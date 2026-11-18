# frigate-detector-edgetpu-yolo9

***NEW 2.1 version as of November 2026 for Frigate v0.18***

First, a brief history of version 1 and 2 of this repo. The official [Frigate](https://github.com/blakeblackshear/frigate) version 0.17 added support to run YOLO v9 models on Google Coral device, based on the model and post-processing code from this repository's version 1.x releases. The [pull request to add YOLO v9 support to the main Frigate repository](https://github.com/blakeblackshear/frigate/pull/21124) was accepted on December 2 2025. Later, in March of 2026 the (Frigate+)[https://docs.frigate.video/plus/] (paid) subscription added YOLO v9s models to run on Coral devices to its offering. In November 2026, a new version 2.0 of this repository introduced a new and incompatible model and detector plugin intended to run on version 0.18 of Frigate. The new v2 model is faster and more accurate than the v.1 model offered by this repo. In comparison with the model currently offered by Frigate+ it is faster but appears to be less accurate when evaluating using publicly available image datasets.

As of November 2026, the model and code from v2 of this repo is a separate project that is not yet incorporated into the official Frigate code. It is not know if or when Frigate maintainers will decide to include this repo's v2 model and code in the official Frigate code and releases.

![abbeyroad-detections](https://github.com/user-attachments/assets/a4dfbe6f-2ed3-43f1-9697-6ea119e463fe)

## Why YOLOv9 and this Plugin?

This repository provides a custom detector plugin for [Frigate](https://github.com/blakeblackshear/frigate), specifically designed to enable the Google Coral Edge TPU to run a **YOLOv9** model to detect objects visible in camera feeds. The plugin handles the necessary post-processing for YOLO model outputs, making them compatible with Frigate's detection pipeline without modifying Frigate's core source code. YOLOv9 model weights can be downloaded here as well, see the release notes and installation instructions.

Frigate is an open-source NVR system that supports various detector acceleration hardware, including the Google Coral Edge TPU. While the default SSD MobileDet model is an effective choice in many systems, running a YOLOv9 model may improve detection accuracy and reduce false positives.

YOLOv9 can run efficiently on the EdgeTPU, and requires a small amount of post-processing by the CPU. This detection method is both high-speed and low-energy, handling approximately 160 detections per second and using only a few Watts of power, while leaving the CPU available for other tasks. Another consideration is minimizing heat produced by the CPU, which can challenge my system's small fan.

There are other YOLO models with incompatible software license terms (eg Ultralytics is licensed with the AGPL). This approach is meant to be used to start with the GPL-licensed YOLOv9 model, which IS compatible with Frigate (also MIT-licensed). For more information, see [this discussion in Frigate's issues](https://github.com/blakeblackshear/frigate/discussions/15630#discussioncomment-11639733).

## Features

*   **Improved Accuracy:** Fewer false positive detections than the v1 model and the default SSD MobileDet object detection model. However, the v2 model is less accurate than the YOLO v9s model available to Frigate+ subscribers.
*   **Google Coral Edge TPU Support:** Optimized for efficient inference on Coral devices. 6ms inference time at 320x320 image size (vs 7ms for the MobileDet model and 9ms for the Frigate+ YOLO v9s model)
*   **Simple Integration:** Adds as a Frigate plugin via a Docker volume mount, no other core Frigate code changes needed.

## Example System Performance

* CPU is Intel 12th Generation i3, 16GB RAM, circa 2022
* Google Coral mini-PCIe card
* 6ms detection speed when running a load test averaging 40 detections per second for images having size 320x320 pixels
* zero skipped frames
* Detector CPU usage varies between 4% and 8%

Regarding CPU usage, all YOLO models (pre YOLO v10) use the CPU during post-processing to decode and de-duplicate the raw results. This model generates box coordinates directly instead of the more complex "DFL" representation of most YOLO models. Box de-duplication is still required as a post-processing step.

## Model Input Size, Detection Sensitivity, and Run Time

There is one version of YOLO v9 model weights available for download that can be used with v2 of this plugin:

* [YOLO v9 "small" "slim" v2.1 320x320 input size](https://github.com/dbro/frigate-detector-edgetpu-yolo9/releases/download/v2.1/yolov9-s-relu6-sl21_320_int8_edgetpu.tflite)

Detection time is approximately 6ms including post-processing steps, allowing 160 detections per second without dropped frames. Note that there can be multiple detections run for a single frame from the video feed, if there are multiple separate areas where motion gets detected.

Technical details:

* Output is two tensors, one for class probabilities and the other for box coordinates in terms of pixel dimensions
* All 288 operations run on the Google Coral TPU
* Zero off-chip memory used
* Model architecture is a "slim" YOLO v9s which reduces the head width from 128 to 64 classes

The ~3ms improvement in detection time comes from the slim head (~2ms), and simplifying the output and reducing post-processing (~1ms).

## Performance Benchmarks

Recommended: **YOLOv9-s ReLU6 slim v2.1** (2-tensor EdgeTPU layout). On a Coral it is about **6.3 ms** end-to-end and beats the v1.0 release on daytime and night Frigate-style crops. It is not as accurate as the current Frigate+ YOLO v9s model, which requires a paid subscription to Frigate+.

| Model | Classes | Detection time | VIRAT wAP | RainSnow wAP |
| ----- | ------- | --------- | --------- | ------------ |
| [YOLOv9-s ReLU6 slim v2.1](https://github.com/dbro/frigate-detector-edgetpu-yolo9/releases/download/v2.1/yolov9-s-relu6-sl21_320_int8_edgetpu.tflite) (**recommended**) | 18 | **6.25 ms** | **74.3** | **25.0** |
| Frigate+ YOLOv9-s (2026-10) | 46 | 9.51 ms | 67.9 | 17.6 |
| [YOLOv9-s ReLU6 slim v2.0](https://github.com/dbro/frigate-detector-edgetpu-yolo9/releases/download/v2.0/yolov9-s-relu6-slim_320_int8_edgetpu.tflite) | 17 | **6.25 ms** | 65.4 | 17.9 |
| [YOLOv9-s ReLU6 v1.0](https://github.com/dbro/frigate-detector-edgetpu-yolo9/releases/download/v1.0/yolov9-s-relu6-best_320_int8_edgetpu.tflite) | 17 | 9.16 ms | 61.4 | 17.1 |
| Frigate+ MobileDet (2026-03) | 46 | **5.70 ms** | 63.7 | 10.5 |
| [SSD/MobileDet](https://github.com/google-coral/test_data/raw/release-frogfish/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite) (Frigate default) | 79 | 6.81 ms | 57.9 | 11.0 |

**Classes** is how many labels the model outputs (17 = the COCO subset in `labels-coco17.txt`; Frigate+ uses its own map; stock MobileDet is 79-id COCO).

**Detection time** is median milliseconds per frame on a Coral EdgeTPU for the full path Frigate cares about: set input + invoke + get outputs + host-side box decode. Lower is better.

**VIRAT wAP** / **RainSnow wAP** are accuracy on Frigate-style detector crops, not full-frame COCO. Each image is a 320×320 region cut the way Frigate 0.18 builds motion regions (cluster → pad → resize). Score is AP at IoU 0.5 (0–100), averaged over person / car / bicycle with fixed weights 0.61 / 0.36 / 0.03 so every row uses the same mix. Higher is better.

Full method, gray/IR stand-in, and more rows: [`benchmark/RESULTS.md`](benchmark/RESULTS.md).

## Data used for training and evaluating the v2 model

Frigate is typically used to monitor surveillance cameras, and there are unfortunately few public data sets that accurately represent these kinds of images. A particular challenge is night time infrared images. The superior accuracy of Frigate+ models is attributed to the proprietary data set contributed by Frigate+ subscribers, used to train these models. While v1 of the model used only COCO data, v2 added two more public data sets and attempted to crop the images to simulate what Frigate would send to an object detector such as Coral. The data sets used to train the v2 model are:

- **VIRAT**: daytime outdoor surveillance ([VIRAT Video Dataset](https://viratdata.org/)). 3000 crops from 233 videos across 9 scenes.
- **RainSnow**: night and dusk traffic cameras in rain and snow ([AAU RainSnow](https://www.kaggle.com/datasets/aalborguniversity/aau-rainsnow), CC BY 4.0). 1588 crops; objects are often small.
- **COCO**: [COCO 2017](https://cocodataset.org/) is a large public object-detection set (about 118k `train2017` images and 5k `val2017` images) with **80** everyday object classes. This model uses COCO data that is filtered to a subset of [17 classes](labels-coco17.txt) and all other labels are dropped. Images that then have no remaining boxes are kept as **background** (negatives). Note that the accuracy measurements above do NOT include COCO data, this is intentional.

## Prerequisites

Before you begin, ensure you have:

* **A running Frigate v0.18 installation:** This plugin has been tested with Frigate **v0.18.2**. This will NOT work with the current (as of November 2026) head version in github, which refactored the detector code files. If there is a future release that includes these changes, then this repo will likely be updated to work with the latest official version of Frigate.
* **Google Coral Edge TPU:** Properly configured and accessible by your Frigate Docker container. Should work with any variation of Google Coral hardware (USB, M2, mPCIe)

## Installation

Follow these steps to integrate the replacement `edgetpu_tfl.py` plugin into your Frigate setup.

### 1. Download the Model Weights, Plugin File and Class Label File

Create a directory on your host system to store the plugin file. For example, you might create `/opt/frigate-plugins/`.

```bash
sudo mkdir -p /opt/frigate-plugins
cd /opt/frigate-plugins
# download weights
sudo wget https://github.com/dbro/frigate-detector-edgetpu-yolo9/releases/download/v2.1/yolov9-s-relu6-sl21_320_int8_edgetpu.tflite
# download plugin
sudo wget https://raw.githubusercontent.com/dbro/frigate-detector-edgetpu-yolo9/main/edgetpu_tfl.py
# download labels
sudo wget https://raw.githubusercontent.com/dbro/frigate-detector-edgetpu-yolo9/main/labels-coco17plus1.txt
```

### 2. Update docker-compose.yml

You need to add a volume mount to your Frigate service in your docker-compose.yml file. This mounts the plugin file into Frigate's detector plugins directory.

Locate your Frigate service definition and add the following two lines in the volumes: section. Adjust /opt/frigate-plugins/edgetpu_tfl.py if you stored the file elsewhere.

The second line to add will make your YOLO model accessible.

```yaml
# ... other services ...
frigate:
  # ... other frigate configurations ...
  volumes:
    # ... existing volumes ...
    - /opt/frigate-plugins/edgetpu_tfl.py:/opt/frigate/frigate/detectors/plugins/edgetpu_tfl.py:ro
    - /opt/frigate-plugins/labels-coco17plus1.txt:/opt/frigate/models/labels-coco17plus1.txt:ro
    - /opt/frigate-plugins/yolov9-s-relu6-sl21_320_int8_edgetpu.tflite:/opt/frigate/models/yolov9-s-relu6-sl21_320_int8_edgetpu.tflite:ro
  # ... rest of frigate service ...
```

After modifying docker-compose.yml, restart your Frigate container:

```bash
docker-compose down
docker-compose up -d
```

### 3. Configure Frigate's config.yml

Now, you need to tell Frigate to use this new detector plugin and your YOLO model.

In your config.yml, in the model: section, add the model_type parameter as shown below and update the model path to point to your YOLO model file that you mounted in the previous step.

```yaml
model:
  model_type: yolo-generic
  labelmap_path: /opt/frigate/models/labels-coco17plus1.txt
  path: /opt/frigate/models/yolov9-s-relu6-sl21_320_int8_edgetpu.tflite
  # Optionally specify the model dimensions (these are the same as Frigate's default 320x320)
  width: 320
  height: 320
detectors:
  coral:
    type: edgetpu
    # ... other detector settings ...
```

### 4. Restart Frigate and Check Performance

Save the Frigate configuration and rexstart Frigate.

* Are there any error messages in Frigate logs during startup?
* What is the inference speed reported in Frigate's System Metrics page?
* What is the level of Detector CPU Usage?
* Are there any skipped frames?
* How many detection errors do you observe? Look for both false positives and false negatives.

## How it Works With Frigate

This modified version of edgetpu_tfl.py replaces the standard version of the plugin and acts as an intermediary. When Frigate requests a detection:

1. Frigate's standard Edge TPU handler passes the image frame to the plugin.
2. The plugin loads your specified YOLO model.
3. It performs inference on the Edge TPU.
4. It then post-processes the raw output from the YOLO model to transform it into the format expected by Frigate (e.g., converting bounding box coordinates, handling confidence scores).
5. The processed detections are returned to Frigate.

Note that Frigate does not automatically send the entire image from the camera to the detector. Instead, it looks for areas in the image that changed from previous frames, and sends only these regions with motion to the detector. There can be more than one region sent per frame from the camera. For example, on a still day, a region might be tightly cropped around a cat walking across a lawn. On a windy autumn day with many leaves or rain creating motion in the overall image, the region of motion could be the entire image or there could be multiple regions sent to the detector. Frigate next resizes the region to match the detection model's input size, then sends it to the detector.

To understand how Frigate uses motion to trigger object detection processing, see [this explanation in the Frigate FAQ](https://docs.frigate.video/configuration/masks/#further-clarification) and a live view of motion regions is in Frigate's "Debug" view in the Settings page..

## Lessons Learned Getting YOLO v9 to Run on Google Coral

It is possible to convert the pre-trained weights for YOLO from PyTorch format to ONNX and then to TFLite and then compile for EdgeTPU, which runs on Google Coral. However, simple conversions of the models available for download perform poorly on the Coral, because it uses operations that are not appropriate for the 8-bit limitation of Google Coral. This leads to a total loss of information and zero detections. The following changes to the YOLO model structure enable the information to be preserved accurately through the sequence of operations run on Google Coral. The goal is to let the Coral perform all the convolution operations, and then send the data for post-processing on the CPU.

* **ReLU6 activation** should be used instead of SiLU activation. It is smaller and faster than SiLU, and quantizes better. Hardswish is another option, which is slower and more accurate.
* **Slim head can achieve comparable accuracy** It has fewer operations and removes the need for off-chip memory use.
* **Send Logit scores** to the CPU for post-processing to transform them into probabilities. The Coral cannot do the sigmoid() transformation.
* **Send simple box coordinates** to the CPU for post-processing instead of "DFL" box location probabilities. Frigate does not need probabilistic box coordinates.
* **Run Non-Maximum Supression (NMS) on the CPU** because the Coral cannot run these operations efficiently.
* **Data size limits are tricky.** For example, using a non-standard order of dimensions (BNC) in the boxes tensor enables it to run entirely on the TPU, but with the standard order (BCN) some operations must either run on the CPU or be done repeatedly on smaller chunks. Coral has a limited amount of on-chip RAM for caching the intermediate results of the operations, and if it needs more it will use the (slower) system RAM.
* **Post-processing takes time** Careful post-processing reduced the overall detection time by about 2ms.

Applying those lessons lets the Google Coral run all of the convolutiuon operations on its TPU for a YOLO v9 "s" slimmed model with size 320x320 pixels. The resulting model uses no off-chip RAM, which would slow down the process.

## Contributing

This project is open-source. If you find issues or have improvements, feel free to open an issue or submit a pull request.

## References

* [https://github.com/blakeblackshear/frigate](https://github.com/blakeblackshear/frigate)
* [https://github.com/WongKinYiu/yolov9/tree/main](https://github.com/WongKinYiu/yolov9/tree/main)
* [https://engineering.cloudflight.io/quantized-yolo-for-edge-solutions](https://engineering.cloudflight.io/quantized-yolo-for-edge-solutions)

## License
This project is licensed under the MIT License. See the LICENSE file for details.
