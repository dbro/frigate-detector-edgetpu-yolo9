# frigate-detector-edgetpu-yolo9

This repository provides a custom detector plugin for Frigate, specifically designed to enable the use of **YOLOv9** models with Google Coral Edge TPUs. This plugin handles the necessary post-processing for YOLO model outputs, making them compatible with Frigate's detection pipeline without modifying Frigate's core source code.

## Why YOLOv9 and this Plugin?

Frigate, an open-source NVR system, supports various detector hardware, including the Google Coral Edge TPU. While the default MobileDet model works well in many case, in some cases YOLOv9 models can often offer improved detection accuracy and reduced false positives.

YOLOv9 can run almost entirely on the TPU of the Google Coral device. This detection method is both high-speed and low-energy, handling approximately 100 detections per second and using only a few Watts of power, while leaving the CPU available for other tasks. Another consideration is minimizing heat produced by the CPU, which can challenge my system's small fan.

However, YOLOv9 is licensed under GPL. To maintain Frigate's MIT license while allowing users to leverage these powerful models with their Coral Edge TPUs, this plugin provides the necessary adapter. By keeping the YOLO-specific processing separate, users can opt-in to using these models by adding this plugin as an optional component. For more information, see [this discussion in Frigate's issues](https://github.com/blakeblackshear/frigate/discussions/15630#discussioncomment-11639733) about how YOLO licensing restricts its inclusion in Frigate's codebase, which uses the MIT license.

## Features

*   **YOLOv9 Compatibility:** Processes output from YOLO models exported for Edge TPU.
*   **Google Coral Edge TPU Support:** Optimized for efficient inference on Coral devices. ~10ms inference time (vs 7ms for the MobileDet model)
*   **Simple Integration:** Adds as a Frigate plugin via a Docker volume mount, no core Frigate code changes needed.
*   **Improved Accuracy:** Fewer false positive detections.

## Caveat

The plugin code does some post-processing on the CPU, so this approach will use the CPU more than the default MobileDet model. However with proper YOLO model selection, the amount of CPU involved does not appear to be a bottleneck (on a very old machine, Intel 3rd generation i7).

## Example System Performance

* Google Coral mini-PCIe card
* 5 cameras
* average 25 camera frames per second
* average 20 detections per second
* zero skipped frames
* 11ms detection speed
* Detector CPU usage varies between 5% and 20%
* CPU is Intel 3rd Generation i7, 8GB RAM


## Prerequisites

Before you begin, ensure you have:

*   **A running Frigate installation:** This plugin has been tested with Frigate **v0.15 and v0.16**.
*   **Google Coral Edge TPU:** Properly configured and accessible by your Frigate Docker container. Should work with any variation of Google Coral hardware (USB, M2, mPCIe)

## Installation

Follow these steps to integrate the `edgetpu_tfl_yolo9.py` plugin into your Frigate setup.

### 1. Get a YOLO Model File for Edge TPU

Exporting YOLO models for Edge TPU involves quantization and conversion. While these instructions are tailored for YOLOv9, this plugin may also work with other YOLO versions (e.g., YOLOv8, YOLO-NAS) that can be exported to the .tflite format. Users are encouraged to test and share their success with other models.

* Download the pytorch version of the yolo model from here: [yolov9t "tiny"](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9t.pt), or [yolov9s "small"](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9s.pt). See the [ultralytics page](https://docs.ultralytics.com/models/yolov9/#performance-on-ms-coco-dataset) for larger versions.
* Create a new Google Colab workbook
* Upload the pytorch .pt file into the Colab workbook
* Create a code block and run these commands:
  * ! pip install ultralytics
  * !yolo mode=export model=yolov9t.pt format=edgetpu imgsz=192
* Run the code block. It will take a few minutes to download and install the ultralytics software, and a few more minutes to translate the .pt file into a file called yolov9t_full_integer_quant_edgetpu.tflite which will be in the Colab folder called yolov9t_saved_model
  * **IMPORTANT** Look at the output of the conversion step and see what it says about the operations being supported by the Edge TPU. Any steps that are not "Mapped to the Edge TPU" will be run on the CPU, which is slower+hotter than if they were to run on the TPU. **LOOK FOR THIS INDICATION OF A PROBLEM**:
```
Model successfully compiled but not all operations are supported by the Edge TPU. A percentage of the model will instead run on the CPU, which is slower. If possible, consider updating your model to use only operations supported by the Edge TPU. For details, visit g.co/coral/model-reqs.
Number of operations that will run on Edge TPU: 233
Number of operations that will run on CPU: 22
```
   * If you see something that indicates a significant number of operations will run on the CPU **re-run the export step again with new settings**. For example, reduce the value for the imgsz parameter to one of these: 320, 288, 256, 224, 192, 160. Or try again with a different .pt file for a different size of the model (instead of "small", try "tiny").
* When you have a model that minimizes CPU operations, download it and copy it to your docker host, to somewhere like /opt/frigate-plugins/

### 2. Download the Plugin File

Create a directory on your host system to store the plugin file. For example, you might create `/opt/frigate-plugins/`.

```bash
sudo mkdir -p /opt/frigate-plugins
cd /opt/frigate-plugins
sudo wget https://raw.githubusercontent.com/dbro/frigate-detector-edgetpu-yolo9/main/edgetpu_tfl_yolo9.py
# Or, if you cloned the repo:
# sudo cp path/to/cloned/repo/edgetpu_tfl_yolo9.py /opt/frigate-plugins/
```

### 3. Update docker-compose.yml

You need to add a volume mount to your Frigate service in your docker-compose.yml file. This mounts the plugin file into Frigate's detector plugins directory.

Locate your Frigate service definition and add the following two lines under the volumes: section. Adjust /opt/frigate-plugins/edgetpu_tfl_yolo9.py if you stored the file elsewhere.

The second line to add will make your YOLO model accessible by the Frigate container by mounting its location on the host.

```yaml
# ... other services ...
frigate:
  # ... other frigate configurations ...
  volumes:
    # ... existing volumes ...
    - /opt/frigate-plugins/edgetpu_tfl_yolo9.py:/opt/frigate/frigate/detectors/plugins/edgetpu_tfl_yolo9.py:ro
    - /opt/frigate-plugins/yolov9t_full_integer_quant_edgetpu.tflite:/opt/frigate/models/yolov9t_full_integer_quant_edgetpu.tflite:ro
  # ... rest of frigate service ...
```

After modifying docker-compose.yml, restart your Frigate container:

```bash
docker-compose down
docker-compose up -d
```

### 4. Configure Frigate's config.yml

Now, you need to tell Frigate to use this new detector plugin and your YOLO model.

In your config.yml, under the detectors: section, replace the existing detector called "edgetpu" with "edgetpu_yolo9" choice, and update the model path to point to your YOLO model file that you mounted in the previous step.

```yaml
detectors:
  coral:
    type: edgetpu_yolo9
    # ... other detector settings ...
  model:
      path: /opt/frigate-plugins/yolov9t_full_integer_quant_edgetpu.tflite # Update this to your model's path
      # Optionally, if your model has specific input dimensions (eg 192x192), uncomment these lines:
      # width: 192
      # height: 192
```

### 5. Restart Frigate and Check Performance

Save the Frigate configuration and rexstart Frigate.

* Are there any error messages in Frigate logs during startup?
* What is the inference speed reported in Frigate's System Metrics page?
* What is the level of Detector CPU Usage?
* Are there any skipped frames?
* How many detection errors do you observe? Look for both false positives and false negatives.

## How it Works

The edgetpu_tfl_yolo9.py plugin acts as an intermediary. When Frigate requests a detection:

1. Frigate's standard Edge TPU handler passes the image frame to the plugin.
2. The plugin loads your specified YOLO model.
3. It performs inference on the Edge TPU.
4. It then post-processes the raw output from the YOLO model to transform it into the format expected by Frigate (e.g., converting bounding box coordinates, handling confidence scores).
5. The processed detections are returned to Frigate.

## Contributing

This project is open-source. If you find issues or have improvements, feel free to open an issue or submit a pull request.

## References

* [https://github.com/blakeblackshear/frigate](https://github.com/blakeblackshear/frigate)
* [https://docs.ultralytics.com/models/yolov9](https://docs.ultralytics.com/models/yolov9)
* [https://github.com/WongKinYiu/yolov9/tree/main](https://github.com/WongKinYiu/yolov9/tree/main)

## License
This project is licensed under the GPL v3.0 License. See the LICENSE file for details.
