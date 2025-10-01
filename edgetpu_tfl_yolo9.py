import logging
import os

import numpy as np
from pydantic import Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig
from frigate.util.model import post_process_yolo

import time

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ModuleNotFoundError:
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate


logger = logging.getLogger(__name__)

DETECTOR_KEY = "edgetpu_yolo9"

class EdgeTpuDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: str = Field(default=None, title="Device Type")


class EdgeTpuTfl(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: EdgeTpuDetectorConfig):
        logger.info(f"Overriding detectors/plugins/edgetpu_tfl.py with host's version intended for Yolo v9")
        device_config = {}
        if detector_config.device is not None:
            device_config = {"device": detector_config.device}

        edge_tpu_delegate = None

        try:
            device_type = (
                device_config["device"] if "device" in device_config else "auto"
            )
            logger.info(f"Attempting to load TPU as {device_type}")
            edge_tpu_delegate = load_delegate("libedgetpu.so.1.0", device_config)
            logger.info("TPU found")
            self.interpreter = Interpreter(
                model_path=detector_config.model.path,
                experimental_delegates=[edge_tpu_delegate],
            )
        except ValueError:
            _, ext = os.path.splitext(detector_config.model.path)

            if ext and ext != ".tflite":
                logger.error(
                    "Incorrect model used with EdgeTPU. Only .tflite models can be used with a Coral EdgeTPU."
                )
            else:
                logger.error(
                    "No EdgeTPU was detected. If you do not have a Coral device yet, you must configure CPU detectors."
                )

            raise

        self.interpreter.allocate_tensors()

        self.tensor_input_details = self.interpreter.get_input_details()
        self.tensor_output_details = self.interpreter.get_output_details()
        #self.model_type = detector_config.model.model_type # is this used anywhere?
        self.model_width = detector_config.model.width
        self.model_height = detector_config.model.height

        self.min_score = 0.4 # yolorest used 0.25
        self.max_detections = 20 # seems to be hard-coded elsewhere, needs to be 20 here

        if len(self.tensor_output_details) == 1:  # generic YOLO has 1 output, SSD has 4
            self.yolo_model = True
        else:
            self.yolo_model = False
            for x in self.tensor_output_details:
                if len(x["shape"]) == 3:
                    self.output_boxes_index = x["index"]
                elif len(x["shape"]) == 1:
                    self.output_count_index = x["index"]

            self.output_class_ids_index = None
            self.output_class_scores_index = None

    def determine_indexes_for_non_yolo_models(self):
        if self.output_class_ids_index is None or self.output_class_scores_index is None:
            for i in range(4):
                index = self.tensor_output_details[i]["index"]
                if index != self.output_boxes_index and index != self.output_count_index:
                    # check if is integer
                    if np.mod(np.float32(self.interpreter.tensor(index)()[0][0]), 1) == 0.0:
                        self.output_class_ids_index = index
                    else:
                        self.output_scores_index = index

    def yolo_preprocess(self, input):
        details = self.tensor_input_details[0]
        input = input.astype('float') / 255
        scale, zero_point = details['quantization']
        input = (input / scale + zero_point).astype(details['dtype'])
        return input

    def yolo_postprocess(self):
        model_width = self.model_width
        model_height = self.model_height
        outputs = []
        for output in self.tensor_output_details:
            x = self.interpreter.get_tensor(output['index'])
            scale, zero_point = output['quantization']
            x = (x.astype(np.float32) - zero_point) * scale
            # Denormalize xywh by image size
            # See https://github.com/ultralytics/ultralytics/pull/1695
            x[:, [0, 2]] *= model_width
            x[:, [1, 3]] *= model_height
            outputs.append(x)

        # Use Frigate's optimized post_process_yolo function
        return post_process_yolo(outputs, model_width, model_height)

    def detect_raw(self, tensor_input):
        if self.yolo_model:
            tensor_input = self.yolo_preprocess(tensor_input)

        self.interpreter.set_tensor(self.tensor_input_details[0]["index"], tensor_input)
        self.interpreter.invoke()

        if self.yolo_model:
            return self.yolo_postprocess()
        else:
            self.determine_indexes_for_non_yolo_models()
            boxes = self.interpreter.tensor(self.tensor_output_details[0]["index"])()[0]
            class_ids = self.interpreter.tensor(self.tensor_output_details[1]["index"])()[0]
            scores = self.interpreter.tensor(self.tensor_output_details[2]["index"])()[0]
            count = int(
                self.interpreter.tensor(self.tensor_output_details[3]["index"])()[0]
            )

        detections = np.zeros((self.max_detections, 6), np.float32)

        for i in range(count):
            if scores[i] < self.min_score:
                break
            if i == self.max_detections:
                logger.info(f"Too many detections ({count})!")
                break
            detections[i] = [
                class_ids[i],
                float(scores[i]),
                boxes[i][0],
                boxes[i][1],
                boxes[i][2],
                boxes[i][3],
            ]

        return detections

