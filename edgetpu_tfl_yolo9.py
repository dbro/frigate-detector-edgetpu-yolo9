import logging
import os

import numpy as np
from pydantic import Field # does YOLO also need Extra ?
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig

import cv2  # for YOLO
import time # for YOLO

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ModuleNotFoundError:
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate


logger = logging.getLogger(__name__)

DETECTOR_KEY = "edgetpu_yolo9"

##### YOLO CODE HERE

USE_YOLO_NMS = True

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Args:
        box1 (numpy array): A numpy array of shape (N, 4) representing N bounding boxes.
        box2 (numpy array): A numpy array of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (numpy array): An NxM numpy array containing the pairwise IoU values for every element in box1 and box2.
    """

    (a1, a2), (b1, b2) = np.split(np.expand_dims(box1, axis=1), 2), np.split(np.expand_dims(box2, axis=1), 2)
    inter = np.clip((np.min(a2, b2) - np.max(a1, b1)), a_min = 0, a_max = None) * 2

    # IoU = inter / (area1 + area2 - inter)
    return inter / (((a2 - a1) * 2)  + ((b2 - b1) * 2) - inter + eps)

def nms(boxes, overlap_threshold=0.5, min_mode=False):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    index_array = scores.argsort()[::-1]
    keep = []
    while index_array.size > 0:
        keep.append(index_array[0])
        x1_ = np.maximum(x1[index_array[0]], x1[index_array[1:]])
        y1_ = np.maximum(y1[index_array[0]], y1[index_array[1:]])
        x2_ = np.minimum(x2[index_array[0]], x2[index_array[1:]])
        y2_ = np.minimum(y2[index_array[0]], y2[index_array[1:]])

        w = np.maximum(0.0, x2_ - x1_ + 1)
        h = np.maximum(0.0, y2_ - y1_ + 1)
        inter = w * h

        if min_mode:
            overlap = inter / np.minimum(areas[index_array[0]], areas[index_array[1:]])
        else:
            overlap = inter / (areas[index_array[0]] + areas[index_array[1:]] - inter)

        inds = np.where(overlap <= overlap_threshold)[0]
        index_array = index_array[inds + 1]
    return keep

def non_max_suppression(
        prediction,
        conf_thres=0.4,
        iou_thres=0.7,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Arguments:
        prediction (numpy array): A numpy array of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, numpy array]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[numpy array]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (
            list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    output = []
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = np.max(prediction[:, 4:mi], axis=1) > conf_thres  # candidates

    # Settings
    min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    prediction = np.transpose(prediction, (0, 2, 1))
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 5))
            v[:, :4] = lb[:, 1:5]  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = np.concatenate((x, v), axis=0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        box, mask = np.array_split(x, [nc], axis=0)
        box, cls = np.array_split(x, [4], axis=1)
        mask = mask.reshape(box.shape[0], 0)

        if multi_label:
            i, j = np.where(cls > conf_thres)
            x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None], mask[i]), 1)
        else:  # best class only
            conf = np.max(cls, axis=1)
            conf = conf.reshape(conf.shape[0], 1)
            j = np.argmax(cls[:, :], axis=1, keepdims=True)
            x = np.concatenate((box, conf, j, mask), axis=1)

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[
                :max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        scores = scores.reshape(scores.shape[0], 1)
        con = np.concatenate((boxes, scores), axis=1)
        keep_boxes = nms(con, iou_thres)  # NMS
        keep_boxes = keep_boxes[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[keep_boxes], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = np.multiply(weights, x[:, :4]).float() / weights.sum(1,
                                                                            keepdim=True)  # merged boxes
            if redundant:
                keep_boxes = keep_boxes[iou.sum(1) > 1]  # require redundancy
        for k in keep_boxes:
            output.append(x[k])
        if (time.time() - t) > time_limit:
            break  # time limit exceeded
    return output

def get_boxes_from_yolo_output(prediction, conf_thres=0.4):
    """
    Get boxes from predictions where confidence threshold is above given limit.

    Arguments:
        prediction (numpy array): A numpy array of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.

    Returns:
        (List[numpy array]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    if isinstance(prediction, (
        list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    output = np.empty(shape=(0,6))
    nc = prediction.shape[1] - 4  # number of classes
    mi = 4 + nc  # mask start index
    xc = np.max(prediction[:, 4:mi], axis=1) > conf_thres  # candidates

    prediction = np.transpose(prediction, (0, 2, 1))

    candidates = prediction[xc]
    candidates[..., :4] = xywh2xyxy(candidates[..., :4])  # xywh to xyxy
    num_of_cars = 0
    for xi, x in enumerate(candidates):
        box, cls = np.array_split(x, [4])

        max_score_class = np.argmax(cls)

        output_row = np.array([box[0], box[1], box[2], box[3], x[4 + max_score_class], max_score_class])
        output = np.vstack((output, output_row))

    # sort based on score
    return output[output[:, 5].argsort()]

##### END OF YOLO CODE SECTION

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

        ### BEGIN YOLO EDITS
        self.min_score = 0.4 # yolorest used 0.25
        self.min_iou = 0.7 # yolorest used 0.45
        self.max_detections = 20 # seems to be hard-coded elsewhere, needs to be 20 here

        if len(self.tensor_output_details) == 1:  # YOLOv8 has 1 output, SSD has 4
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
        y = []
        for output in self.tensor_output_details:
            x = self.interpreter.get_tensor(output['index'])

            scale, zero_point = output['quantization']
            x = (x.astype(np.float32) - zero_point) * scale  # re-scale

            # Denormalize xywh by image size. See https://github.com/ultralytics/ultralytics/pull/1695
            # xywh are normalized in TFLite/EdgeTPU to mitigate quantization error of integer models
            x[:, [0, 2]] *= model_width
            x[:, [1, 3]] *= model_height
            y.append(x)

        if USE_YOLO_NMS:
            detections = non_max_suppression(y, conf_thres=self.min_score, iou_thres=self.min_iou)
        else:
            detections = get_boxes_from_yolo_output(y, conf_thres=self.min_score)
        
        boxes = []
        class_ids = []
        scores = []

        for det in detections:
            boxes.append(
                [det[1] / model_height,
                det[0] / model_width,
                det[3] / model_height,
                det[2] / model_width])
            scores.append(det[4])
            class_ids.append(det[5])

        return boxes, scores, class_ids, len(boxes)

    ### END YOLO EDITS

    def detect_raw(self, tensor_input):
        if self.yolo_model: # YOLO
            tensor_input = self.yolo_preprocess(tensor_input)

        self.interpreter.set_tensor(self.tensor_input_details[0]["index"], tensor_input)
        self.interpreter.invoke()

        if self.yolo_model: # YOLO
            boxes, scores, class_ids, count = self.yolo_postprocess()
        else:
            self.determin_indexes_for_non_yolo_models()
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

