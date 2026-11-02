# Benchmarks for frigate-detector-edgetpu-yolov9

This folder has two things:

1. **[RESULTS.md](RESULTS.md)**: accuracy and speed of this release's models against Frigate's stock Coral model and
   Frigate+ models, on Frigate-style 320×320 camera crops (daytime VIRAT, night RainSnow, and RainSnow in gray).
   Start there to pick a model.
2. A small Docker harness (`benchmark.py`, `detect_image.py`, `frigate_shim.py`) that measures COCO mAP and
   Coral speed, or runs one image, using a copy of the patched EdgeTPU detector's post-processing.

## Models

| model | box output | notes |
|---|---|---|
| slim r6, 2t (**this release**) | `(1,2100,4)` ltrb distances + `(1,2100,17)` class logits | fastest YOLO (6.25 ms e2e on Coral); preferred for Frigate |
| slim hs | same slim head (`distlite_cls64`), hard-swish activations | more accurate at night / in gray, about +2.9 ms per frame |
| v1.0 / v1.5 (legacy) | `(1,2100,64)` DFL + `(1,2100,17)` class logits | the original DualDDetect head with DFL box decoding |

`frigate_shim.py` picks the box decode by the box tensor's channel count: 64 channels → DFL (the legacy models,
unchanged), 4 channels → distlite (`max(d, 0)` ltrb distances, same anchors and strides). The log line
`YOLO box decode: distlite (...)` or `YOLO box decode: dfl (...)` at start-up shows which one was chosen.

The scripts treat a model as YOLO when its file name contains `yolo`, and as SSD otherwise. If your file name does
not contain `yolo`, pass `--model_type yolo` (or `--model_type ssd`).

The numbers in RESULTS.md come from a separate harness that replays Frigate 0.18's region cropping, pre- and
post-processing on camera data sets. The COCO commands below give a second, simpler check (mAP on full COCO val
images, 17 classes).

## Requirements

In addition to a system with docker installed, download the COCO validation images and labels here:

* [COCO validation images](http://images.cocodataset.org/zips/val2017.zip)
* [labels](https://huggingface.co/datasets/merve/coco/blob/main/annotations/instances_train2017.json)

The threshold for minimum score is 0.4, and the NMS threshold is 0.4. mAP50 counts a detection as correct only if
the predicted box overlaps the real object by at least 50 % (IoU ≥ 0.5).

## Example command to measure speed and accuracy using COCO validation images

Make sure there are no other processes running that use the Coral device, ie. stop the Frigate container.

```
docker build -t benchmark .

docker run -it --rm \
  --privileged \
  --device /dev/apex_0 \
  -v $(pwd):/app \
  -v /path/to/your/models:/models \
  -v /path/to/coco:/coco \
  benchmark \
  --model /models/yolov9-s-relu6-slim_320_int8_edgetpu.tflite \
  --size 320 \
  --coco_ann /coco/instances_val2017.json \
  --coco_img_dir /coco/val2017/
```

To benchmark a legacy DFL model, pass its file instead, for example
`--model /models/yolov9-s-relu6-best_320_int8_edgetpu.tflite`. Add `--limit 500` for a quick run.

## Example command to run inference on a single image

Make sure there are no other processes running that use the Coral device, ie. stop the Frigate container.

```
docker build -t benchmark .

docker run -it --rm \
  --entrypoint python \
  --privileged \
  --device /dev/apex_0 \
  -v $(pwd):/app \
  -v /path/to/your/models:/models \
  benchmark \
  detect_image.py \
  --model /models/yolov9-s-relu6-slim_320_int8_edgetpu.tflite \
  --size 320 \
  --labels /models/labels-coco17.txt \
  --image AbbeyRoad.jpg \
  --output AbbeyRoad-detected-yolov9-s-relu6-slim-320.jpg
```
