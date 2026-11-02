import cv2
import numpy as np
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import time
from frigate_shim import EdgeTpuTfl, EdgeTpuDetectorConfig, ModelConfig, ModelTypeEnum

# --- MAPPING ---
# Frigate YOLOv9 subset (0-16) -> COCO Original ID (1-90)
FRIGATE_TO_COCO = {
    0: 1,  # person
    1: 2,  # bicycle
    2: 3,  # car
    3: 4,  # motorcycle
    4: 5,  # airplane
    5: 6,  # bus
    6: 7,  # train
    7: 8,  # truck
    8: 9,  # boat
    9: 16, # bird
    10: 17, # cat
    11: 18, # dog
    12: 19, # horse
    13: 20, # sheep
    14: 21, # cow
    15: 22, # elephant
    16: 23  # bear
}
# List of COCO IDs we care about for the final score
TARGET_COCO_IDS = list(FRIGATE_TO_COCO.values())

def main(args):
    # 1. Load COCO Ground Truth
    print(f"Loading COCO annotations from {args.coco_ann}...")
    coco_gt = COCO(args.coco_ann)
    
    # Get all image IDs
    img_ids = coco_gt.getImgIds()
    if args.limit:
        img_ids = img_ids[:args.limit]
        print(f"Limiting to first {args.limit} images.")

    # 2. Initialize Model
    if args.model_type == "auto":  # guess from the file name
        model_type = ModelTypeEnum.yologeneric if "yolo" in args.model.lower() else ModelTypeEnum.ssd
    else:
        model_type = ModelTypeEnum.yologeneric if args.model_type == "yolo" else ModelTypeEnum.ssd
    
    config = EdgeTpuDetectorConfig(
        model=ModelConfig(
            path=args.model,
            width=args.size,
            height=args.size,
            model_type=model_type
        )
    )
    
    detector = EdgeTpuTfl(config)
    print(f"Model loaded. Type: {model_type}, Input Size: {args.size}x{args.size}")

    results = []
    
    # Timings
    tpu_times = []
    nms_times = []
    overhead_times = [] # Capture time spent in int8 conversion/python overhead

    print(f"Starting inference on {len(img_ids)} images...")
    
    for img_id in img_ids:
        # Load Image
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(args.coco_img_dir, img_info['file_name'])
        
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not read {img_path}")
            continue

        original_h, original_w = frame.shape[:2]

        # --- PREPROCESSING ---
        # Resize and ensure RGB (standard Frigate flow)
        input_frame = cv2.resize(frame, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(input_frame, axis=0)

        # --- INFERENCE ---
        # We wrap the call to measure total Python time vs Internal TPU time
        start_wall = time.perf_counter()
        
        # pass uint8; the shim handles the int8 cast if needed
        detections = detector.detect_raw(input_tensor) 
        
        end_wall = time.perf_counter()
        
        # Record Timings
        # detector.last_inference_time is pure TPU invoke time
        # detector.last_postprocess_time is pure NMS/decoding time
        tpu_ms = detector.last_inference_time * 1000
        nms_ms = detector.last_postprocess_time * 1000
        wall_ms = (end_wall - start_wall) * 1000
        
        # "Overhead" is the time spent transforming input (int8 cast) + function call overhead
        overhead_ms = max(0, wall_ms - tpu_ms - nms_ms)
        
        tpu_times.append(tpu_ms)
        nms_times.append(nms_ms)
        overhead_times.append(overhead_ms)

        # --- FORMATTING RESULTS ---
        for det in detections:
            class_id, score, ymin, xmin, ymax, xmax = det
            
            if score < 0.05: continue # Slightly higher thresh for Benchmark speed
            
            # Map Model Class ID to COCO Class ID
            if int(class_id) not in FRIGATE_TO_COCO:
                continue
                
            coco_category_id = FRIGATE_TO_COCO[int(class_id)]

            # Denormalize
            abs_x = xmin * original_w
            abs_y = ymin * original_h
            abs_w = (xmax - xmin) * original_w
            abs_h = (ymax - ymin) * original_h

            result = {
                "image_id": img_id,
                "category_id": coco_category_id,
                "bbox": [abs_x, abs_y, abs_w, abs_h],
                "score": float(score)
            }
            results.append(result)

    print(f"\nInference Complete.")
    print(f"Avg Preprocess/Cast:    {np.mean(overhead_times):.2f} ms (Includes int8 conversion if active)")
    print(f"Avg TPU Inference:      {np.mean(tpu_times):.2f} ms")
    print(f"Avg CPU Postprocess:    {np.mean(nms_times):.2f} ms")
    print(f"--------------------------------")
    print(f"Avg Total Latency:      {np.mean(overhead_times) + np.mean(tpu_times) + np.mean(nms_times):.2f} ms")

    # 3. Calculate mAP
    if not results:
        print("No detections made.")
        return

    print("Evaluating mAP...")
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # CRITICAL: Filter evaluation to ONLY the classes supported by the model
    coco_eval.params.catIds = TARGET_COCO_IDS
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    parser.add_argument("--size", type=int, default=320, help="Model input size (e.g. 320, 512)")
    parser.add_argument("--coco_ann", required=True, help="Path to instances_val2017.json")
    parser.add_argument("--coco_img_dir", required=True, help="Path to val2017 image directory")
    parser.add_argument("--model_type", choices=["auto", "yolo", "ssd"], default="auto",
                        help="auto = YOLO if the file name contains 'yolo', else SSD")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images (for quick testing)")
    args = parser.parse_args()
    main(args)
