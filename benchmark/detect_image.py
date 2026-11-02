import argparse
import cv2
import numpy as np
import os
from frigate_shim import EdgeTpuTfl, EdgeTpuDetectorConfig, ModelConfig, ModelTypeEnum

def load_label_map(label_path):
    """Loads a label map file (one label per line)."""
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels

def main(args):
    # 1. Load Labels
    if args.labels:
        print(f"Loading labels from: {args.labels}")
        labels = load_label_map(args.labels)
    else:
        print("Warning: No label file provided. Using generic ID numbers.")
        labels = []

    # 2. Initialize Detector
    print(f"Loading model: {args.model}")
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

    # 3. Load and Preprocess Image
    print(f"Reading image: {args.image}")
    original_img = cv2.imread(args.image)
    if original_img is None:
        print(f"Error: Could not read image {args.image}")
        return

    # Resize to model input size (Squashing)
    input_img = cv2.resize(original_img, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
    # Convert BGR to RGB
    input_tensor = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Check Input Format
    input_details = detector.tensor_input_details[0]
    if input_details['dtype'] == np.int8 and input_tensor.dtype == np.uint8:
        scale, zero_point = input_details['quantization']
        print(f"Model expects int8. Scale: {scale}, Zero Point: {zero_point}")
        input_tensor = (input_tensor.view(np.int8) + int(zero_point)).astype(np.int8)

    # 4. Run Inference
    print("Running inference...")
    detections = detector.detect_raw(input_tensor)

    # 5. Process Results
    count = 0
    h, w = original_img.shape[:2]
    
    raw_detection_count = 0 if detections is None else len(detections)
    print(f"\ndetector returned {raw_detection_count} raw results before filtering and NMS")

    print(f"\n{'CLASS':<15} {'SCORE':<8} {'BOX (x1, y1, x2, y2)':<20}")
    print("-" * 50)

    if detections is not None:
      for det in detections:
        class_id, score, ymin, xmin, ymax, xmax = det
        
        if score < args.thresh:
            continue
            
        count += 1
        class_id = int(class_id)
        
        # Determine Label Name
        if 0 <= class_id < len(labels):
            label_name = labels[class_id]
        else:
            label_name = f"ID {class_id}"

        # Convert normalized coordinates back to original image size
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)

        # Print to Console
        print(f"{label_name:<15} {score:.1%}     [{x1}, {y1}, {x2}, {y2}]")

        # Draw on Image
        color = (0, 255, 0) # Green
        cv2.rectangle(original_img, (x1, y1), (x2, y2), color, 2)
        
        # Label text
        text = f"{label_name} {score:.0%}"
        
        # Text background for readability
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(original_img, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
        cv2.putText(original_img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # 6. Save Output
    cv2.imwrite(args.output, original_img)
    print(f"\nDone. Found {count} objects. Saved visualization to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--labels", required=True, help="Path to label map text file")
    parser.add_argument("--output", default="output.jpg", help="Path to save result")
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--model_type", choices=["auto", "yolo", "ssd"], default="auto",
                        help="auto = YOLO if the file name contains 'yolo', else SSD")
    parser.add_argument("--thresh", type=float, default=0.4, help="Confidence threshold")
    args = parser.parse_args()
    main(args)
