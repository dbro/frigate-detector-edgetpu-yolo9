import logging
import os
import numpy as np
import time
import cv2
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field

# --- MOCKING FRIGATE DEPENDENCIES ---
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class ModelTypeEnum(str, Enum):
    ssd = "ssd"
    yologeneric = "yologeneric"

class ModelConfig(BaseModel):
    path: str
    width: int
    height: int
    model_type: ModelTypeEnum

class EdgeTpuDetectorConfig(BaseModel):
    type: Literal["edgetpu"] = "edgetpu"
    device: Optional[str] = None
    model: ModelConfig

# --- HELPER: YOLO POST PROCESS (Added to replace missing import) ---
def post_process_yolo(outputs, w, h):
    """
    Standard YOLO post-processing for single-head outputs.
    Expects outputs[0] to be (1, N, C+4) or (1, C+4, N).
    Coordinates are expected to be in pixels (already denormalized).
    """
    prediction = outputs[0]
    
    # Remove batch dimension -> (N, C+4) or (C+4, N)
    if prediction.ndim == 3:
        prediction = prediction[0]
    
    # Transpose if necessary so that shape is (N, C+4)
    # Heuristic: usually Number of anchors (N) >> Number of channels (C)
    if prediction.shape[0] < prediction.shape[1]:
        prediction = prediction.T
        
    # Now prediction is (N, C+4). Columns: [x, y, w, h, (optional obj), class_scores...]
    boxes = prediction[:, :4]
    scores_data = prediction[:, 4:]
    
    # Identify scores
    if scores_data.shape[1] > 0:
        # Simple Logic: Max of class scores
        # Note: If there is an explicit objectness column (like YOLOv5's 5th column), 
        # this logic assumes it's handled or we just take the max class probability.
        # For a generic shim, simply taking argmax of the remaining columns is usually sufficient.
        class_ids = np.argmax(scores_data, axis=1)
        max_scores = np.max(scores_data, axis=1)
    else:
        return np.zeros((20, 6), np.float32)

    # Threshold Filter
    conf_thresh = 0.4
    mask = max_scores >= conf_thresh
    
    if not np.any(mask):
        return np.zeros((20, 6), np.float32)
    
    boxes = boxes[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]
    
    # Convert xywh (center) to xyxy (corners)
    # x, y are center. w, h are full width/height
    x_c, y_c, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_c - bw / 2
    y1 = y_c - bh / 2
    x2 = x_c + bw / 2
    y2 = y_c + bh / 2
    
    # Stack for NMS
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    
    # NMS
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xyxy.tolist(),
        scores=max_scores.tolist(),
        score_threshold=conf_thresh,
        nms_threshold=0.4
    )
    
    detections = np.zeros((20, 6), np.float32)
    
    if len(indices) > 0:
        indices = np.array(indices).flatten()
        indices = indices[:20] # Limit to max detections
        
        count = 0
        for idx in indices:
            # Output format: [class_id, score, ymin, xmin, ymax, xmax]
            # Must normalize coordinates to 0-1 for Frigate
            b = boxes_xyxy[idx]
            
            detections[count] = [
                class_ids[idx],
                max_scores[idx],
                b[1] / h, # ymin
                b[0] / w, # xmin
                b[3] / h, # ymax
                b[2] / w, # xmax
            ]
            count += 1
            
    return detections

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ModuleNotFoundError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter, load_delegate
    except ImportError:
        logger.error("Could not import tflite_runtime or tensorflow.lite")
        raise

# --- DETECTOR CLASS ---

class EdgeTpuTfl:
    def __init__(self, detector_config: EdgeTpuDetectorConfig):
        logger.info(f"Initializing Detector: {detector_config.model.path}")
        
        device_config = {}
        if detector_config.device is not None:
            device_config = {"device": detector_config.device}

        edge_tpu_delegate = None
        try:
            edge_tpu_delegate = load_delegate("libedgetpu.so.1.0", device_config)
            self.interpreter = Interpreter(
                model_path=detector_config.model.path,
                experimental_delegates=[edge_tpu_delegate],
            )
        except Exception as e:
            logger.error(f"Error loading EdgeTPU: {e}")
            raise

        self.interpreter.allocate_tensors()

        self.tensor_input_details = self.interpreter.get_input_details()
        self.tensor_output_details = self.interpreter.get_output_details()
        self.model_width = detector_config.model.width
        self.model_height = detector_config.model.height

        # Check if model expects int8 input
        dtype = self.tensor_input_details[0]['dtype']
        if dtype == np.int8:
            logger.info(f"Model requires int8 input. Using XOR method.")

        self.min_score = 0.4
        self.max_detections = 20

        model_type = detector_config.model.model_type
        self.yolo_model = model_type == ModelTypeEnum.yologeneric
        
        self.last_inference_time = 0
        self.last_postprocess_time = 0

        if self.yolo_model:
            logger.info(f"Preparing YOLO postprocessing for {len(self.tensor_output_details)}-tensor output")
            if len(self.tensor_output_details) in [2,3]:
                self._generate_anchors_and_strides()
                self.min_logit_value = np.log(self.min_score / (1 - self.min_score))
                self.reg_max = 16 
                self.project = np.arange(self.reg_max, dtype=np.float32)
                
                self.output_boxes_index = None
                self.output_classes_index = None
                
                # Box tensor: C == 64 (DFL, 4 x 16 bins) is preferred; otherwise
                # C == 4 ("distlite": ltrb distances regressed directly, in stride units).
                # The class tensor is the other 3D output with C > 1.
                for box_channels in (64, 4):
                    for i, x in enumerate(self.tensor_output_details):
                        if len(x["shape"]) == 3 and x["shape"][2] == box_channels:
                            self.output_boxes_index = i
                            break
                    if self.output_boxes_index is not None:
                        break
                for i, x in enumerate(self.tensor_output_details):
                    if i != self.output_boxes_index and len(x["shape"]) == 3 and x["shape"][2] > 1:
                        self.output_classes_index = i
                
                if self.output_boxes_index is None or self.output_classes_index is None:
                    self.output_classes_index = 0 
                    self.output_boxes_index = 1 

                boxes_shape = tuple(int(v) for v in self.tensor_output_details[self.output_boxes_index]["shape"])
                scores_shape = tuple(int(v) for v in self.tensor_output_details[self.output_classes_index]["shape"])
                self.box_decode = "distlite" if boxes_shape[-1] == 4 else "dfl"
                logger.info(f"YOLO box decode: {self.box_decode} (boxes tensor {boxes_shape}, class tensor {scores_shape})")
        else:
            for x in self.tensor_output_details:
                if len(x["shape"]) == 3:
                    self.output_boxes_index = x["index"]
                elif len(x["shape"]) == 1:
                    self.output_count_index = x["index"]
            self.output_class_ids_index = None
            self.output_class_scores_index = None

    def _generate_anchors_and_strides(self):
        all_anchors = []
        all_strides = []
        strides = (8, 16, 32) 

        for stride in strides:
            feat_h, feat_w = self.model_height // stride, self.model_width // stride
            grid_y, grid_x = np.meshgrid(
                np.arange(feat_h, dtype=np.float32),
                np.arange(feat_w, dtype=np.float32),
                indexing='ij'
            )
            grid_coords = np.stack((grid_x.flatten(), grid_y.flatten()), axis=1)
            anchor_points = grid_coords + 0.5
            all_anchors.append(anchor_points)
            all_strides.append(np.full((feat_h * feat_w, 1), stride, dtype=np.float32))

        self.anchors = np.concatenate(all_anchors, axis=0)
        self.anchor_strides = np.concatenate(all_strides, axis=0)

    def determine_indexes_for_non_yolo_models(self):
        if self.output_class_ids_index is None or self.output_class_scores_index is None:
            for i in range(4):
                index = self.tensor_output_details[i]["index"]
                if index != self.output_boxes_index and index != self.output_count_index:
                    if np.mod(np.float32(self.interpreter.tensor(index)()[0][0]), 1) == 0.0:
                        self.output_class_ids_index = index
                    else:
                        self.output_scores_index = index

    def detect_raw(self, tensor_input):
        # Apply int8 transformation if needed
        if tensor_input.dtype == np.uint8 and self.tensor_input_details[0]['dtype'] == np.int8:
            tensor_input = np.bitwise_xor(tensor_input, 128).view(np.int8)

        start_inference = time.perf_counter()
        self.interpreter.set_tensor(self.tensor_input_details[0]["index"], tensor_input)
        self.interpreter.invoke()
        
        if self.yolo_model:
            # Single Tensor YOLO Path
            if len(self.tensor_output_details) == 1:
                end_inference = time.perf_counter()
                self.last_inference_time = end_inference - start_inference

                outputs = []
                for output in self.tensor_output_details:
                    x = self.interpreter.get_tensor(output['index'])
                    scale, zero_point = output['quantization']
                    x = (x.astype(np.float32) - zero_point) * scale
                    # Denormalize xywh by image size (pixels)
                    x[:, [0, 2]] *= self.model_width
                    x[:, [1, 3]] *= self.model_height
                    outputs.append(x)
                
                # Use the helper function defined above
                detections = post_process_yolo(outputs, self.model_width, self.model_height)
                self.last_postprocess_time = time.perf_counter() - end_inference
                return detections

            else:
                # Multi-tensor YOLO model with (non-standard B(H*W)C output format).
                # (the comments indicate the shape of tensors, using "2100" as the anchor count
                # corresponding to an image size of 320x320, "NC" as number of classes,
                # "N" as the count that survive after min-score filtering)
                # TENSOR A) class scores (1, 2100, NC) where NC is the count of classes, and the score values are logits
                # TENSOR B) box coordinates (1, 2100, 64) encoded as dfl scores,
                #           or (1, 2100, 4) ltrb distances in stride units for "distlite" models
                # Note that the logit values in tensor (A) should be clamped to the range [-4,+4]
                # to preserve precision in the useful range between ~2% and 98%
                # and because NMS requires the min_score parameter to be >= 0
                end_inference = time.perf_counter()
                self.last_inference_time = end_inference - start_inference

                # Start by getting the quantization scaling info to translate thresholds into quantized amounts
                # but don't dequantize scores data yet, wait until the low-confidence candidates
                # are filtered out from the overall result set. This reduces the work and makes post-processing faster.
                # this method works with raw quantized numbers when possible, which relies on the
                # value of the scale factor to be >0. This speeds up max and argmax operations.
                # Get max confidence for each detection and create the mask to filter low confidence detections
                detections = np.zeros((self.max_detections, 6), np.float32) # initialize zero results
                scores_details = self.tensor_output_details[self.output_classes_index]
                scores_scale, scores_zero_point = scores_details['quantization']
                min_score_quantized = int((self.min_logit_value / scores_scale) + scores_zero_point) # constant
                scores_output_quantized = self.interpreter.get_tensor(scores_details['index'])[0] # (2100, NC)
                max_scores_quantized = np.max(scores_output_quantized, axis=1)  # (2100,)
                mask = max_scores_quantized >= min_score_quantized  # (2100,)

                if not np.any(mask):
                    return detections # empty results

                max_scores_filtered_plus4 = ((max_scores_quantized[mask] - scores_zero_point) * scores_scale) + 4.0  # (N,1) shifted logit values
                scores_output_quantized_filtered = scores_output_quantized[mask]

                # dequantize boxes. NMS needs them to be in float format
                boxes_details = self.tensor_output_details[self.output_boxes_index]
                boxes_scale, boxes_zero_point = boxes_details['quantization']
                # remove candidates with probabilities < threshold
                boxes_output_quantized_filtered = (self.interpreter.get_tensor(boxes_details['index'])[0])[mask]  # (N, 64) or (N, 4)
                boxes_output_filtered = (boxes_output_quantized_filtered.astype(np.float32) - boxes_zero_point) * boxes_scale

                # 2. Decode to distances (ltrb)
                if self.box_decode == "distlite":
                    # distances are regressed directly; only clamp negatives (same as training/eval)
                    distances = np.maximum(boxes_output_filtered, 0.0)  # (N, 4)
                else:
                    dfl_distributions = boxes_output_filtered.reshape(-1, 4, self.reg_max)  # (N, 4, 16)

                    # Softmax over the 16 bins
                    dfl_max = np.max(dfl_distributions, axis=2, keepdims=True)
                    dfl_exp = np.exp(dfl_distributions - dfl_max)
                    dfl_probs = dfl_exp / np.sum(dfl_exp, axis=2, keepdims=True)  # (N, 4, 16)

                    # Weighted sum: (N, 4, 16) * (16,) -> (N, 4)
                    distances = np.einsum('pcr,r->pc', dfl_probs, self.project)

                # Calculate box corners in pixel coordinates
                anchors_filtered = self.anchors[mask]
                anchor_strides_filtered = self.anchor_strides[mask]
                x1y1 = (anchors_filtered - distances[:, [0, 1]]) * anchor_strides_filtered  # (N, 2)
                x2y2 = (anchors_filtered + distances[:, [2, 3]]) * anchor_strides_filtered  # (N, 2)
                boxes_filtered_decoded = np.concatenate((x1y1, x2y2), axis=-1)  # (N, 4)

                # 9. Apply NMS. Use logit scores here to defer sigmoid() until after filtering out redundant boxes
                indices = cv2.dnn.NMSBoxes(
                    bboxes=boxes_filtered_decoded,
                    scores=max_scores_filtered_plus4, # logit scores shifted to be non-negative to allow min_score <50% or logit<0
                    score_threshold=(self.min_logit_value + 4.0), # cv2 asserts the min_score must be > 0 which means >50% probability
                    nms_threshold=0.4 # should this be a model config setting?
                )
                #logger.info(f"NMS kept {indices.shape} boxes out of {np.sum(mask)}")
                num_detections = len(indices)
                if num_detections == 0:
                    return detections # empty results

                nms_indices = np.array(indices, dtype=np.int32).ravel()  # or .flatten()
                if num_detections > self.max_detections:
                    nms_indices = nms_indices[:self.max_detections]
                    num_detections = self.max_detections
                kept_logits_quantized = scores_output_quantized_filtered[nms_indices]
                class_ids_post_nms = np.argmax(kept_logits_quantized, axis=1)

                # Extract the final boxes and scores using fancy indexing
                final_boxes = boxes_filtered_decoded[nms_indices]
                final_scores_logits = max_scores_filtered_plus4[nms_indices] - 4.0 # Unshifted logits

                # Detections array format: [class_id, score, ymin, xmin, ymax, xmax]
                detections[:num_detections, 0] = class_ids_post_nms
                detections[:num_detections, 1] = 1.0 / (1.0 + np.exp(-final_scores_logits)) # sigmoid
                detections[:num_detections, 2] = final_boxes[:, 1] / self.model_height
                detections[:num_detections, 3] = final_boxes[:, 0] / self.model_width
                detections[:num_detections, 4] = final_boxes[:, 3] / self.model_height
                detections[:num_detections, 5] = final_boxes[:, 2] / self.model_width

                self.last_postprocess_time = time.perf_counter() - end_inference
                return detections
        else:
            # SSD Model
            
            end_inference = time.perf_counter()
            self.last_inference_time = end_inference - start_inference

            self.determine_indexes_for_non_yolo_models()
            boxes = self.interpreter.tensor(self.tensor_output_details[0]["index"])()[0]
            class_ids = self.interpreter.tensor(self.tensor_output_details[1]["index"])()[0]
            scores = self.interpreter.tensor(self.tensor_output_details[2]["index"])()[0]
            count = int(self.interpreter.tensor(self.tensor_output_details[3]["index"])()[0])

            detections = np.zeros((self.max_detections, 6), np.float32)

            for i in range(count):
                if scores[i] < self.min_score:
                    break
                if i == self.max_detections:
                    break
                detections[i] = [
                    class_ids[i],
                    float(scores[i]),
                    boxes[i][0],
                    boxes[i][1],
                    boxes[i][2],
                    boxes[i][3],
                ]
            self.last_postprocess_time = time.perf_counter() - end_inference
            return detections
