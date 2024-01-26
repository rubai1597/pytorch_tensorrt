# <div align=center>Installation</div>
## Install TensorRT from the Python Package.
This step referred to [nvidia tensorrt quide](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#installing-pip) and [pytorch tensorrt guide](https://pytorch.org/TensorRT/getting_started/installation.html#installation).

### 1. create Anaconda environment
```bash
conda create -n torch_tensorrt python=3.10 -y
conda activate torch_tensorrt
```

### 2. Install tensorrt

```bash
pip install packaging==23.2
pip install tensorrt==8.6.1
pip install tensorrt_libs==8.6.1
pip install tensorrt_bindings==8.6.1
```

### 3. Install torch-tensorrt and torchvision
pytorch(2.0.1) will be installed together when torch-tensorrt is installed. Additional installation is only required for torchvision.

```bash
pip install torch-tensorrt==1.4.0
pip install torchvision==0.15.2
```
# <div align=center>Test</div>

## Test PyTorch Models Converted to TensorRT.
Download image from [url](https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg).

```bash
wget https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg -O golden_retrieval.jpg
python test_torch_tensorrt.py --model resnet50 --test_img golden_retrieval.jpg
```

## Inference Speed Comparison

With RTX4080 (10000 iterations),

| Model Name   | Speed<br><sup>PyTorch<br>(ms)   | Speed<br><sup>TensorRT<br>(ms)      |
| ------------ | ------------------------------- | ----------------------------------- |
| resnet18     | 1.6425                          | 0.1688                              |
| resnet34     | 2.8721                          | 0.2669                              |
| resnet50     | 4.0834                          | 0.2989                              |
| resnet101    | 8.0887                          | 0.6094                              |
| resnet152    | 11.9056                         | 0.8631                              |

# <div align=center>Inference with onnxruntime</div>
Converting some PyTorch models to TensorRT can be challenging. Instead, if you convert them to ONNX, you can inference them queickly using the **onnxruntime** package. 

Empirically, we found that onnxruntime is slower than torch_tensorrt.

## Install onnxruntime-gpu
You should install onnxruntime-gpu, not onnxruntime.
If you installed onnxruntime already, uninstall it and install onnxruntime-gpu.
```bash
# necessary if you installed onnxruntime
pip uninstall onnxruntime
pip install onnxruntime-gpu==1.16.3
```

## Inference ONNX model with onnxruntime
```bash
wget https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg -O golden_retrieval.jpg
python test_onnxruntime_tensorrt.py --model resnet50 --test_img golden_retrieval.jpg
```

## Inference Speed Comparison

With RTX4080 (10000 iterations),

| Model Name   | Speed<br><sup>PyTorch<br>(ms)   | Speed<br><sup>TensorRT<br>(ms)      |
| ------------ | ------------------------------- | ----------------------------------- |
| resnet18     | 1.6573                          | 0.3272                              |
| resnet34     | 2.9409                          | 0.4504                              |
| resnet50     | 4.2568                          | 0.4888                              |
| resnet101    | 8.1113                          | 0.7973                              |
| resnet152    | 12.1374                         | 1.0619                              |

## Test with YOLOv8
First, install ultralytics package
```bash
pip install ultralytics==8.1.6
```

Test with yolov8n. It would be a quite slow when the model is first converted.

ultralytics can convert torch models into tensorrt, but we will use onnxruntime inference session for versatility, such as [onnx-modifier](https://github.com/ZhangGe6/onnx-modifier).

Ultralytics have capability of converting PyTorch models into TensorRT format. However, for versatility, including the use of tools like ONNX-Modifier, we adopt ONNXRuntime inference sessions.

Remove comment from line 57 to 88 If you want to try convert yolov8 torchscript model into tensorrt.

```bash
# Optional if you want to ignore ultralytics' logs in inference.
export YOLO_VERBOSE=false
python test_yolov8_onnxruntime.py --model yolov8n
```

With RTX4080 (10000 iterations),

<details open><summary>Detection</summary>

| Model Name   | Speed<br><sup>PyTorch<br>(ms)   | Speed<br><sup>TensorRT<br>(ms)      |
| ------------ | ------------------------------- | ----------------------------------- |
| yolov8n      | 4.1913                          | 1.3731                              |
| yolov8s      | 4.3005                          | 1.5856                              |
| yolov8m      | 5.2719                          | 2.1509                              |
| yolov8l      | 7.8529                          | 2.9957                              |
| yolov8x      | 10.8270                         | 3.6225                              |

</details>

<details open><summary>Pose</summary>

| Model Name   | Speed<br><sup>PyTorch<br>(ms)   | Speed<br><sup>TensorRT<br>(ms)      |
| ------------ | ------------------------------- | ----------------------------------- |
| yolov8n-pose | 4.9000                          | 1.3158                              |
| yolov8s-pose | 5.0468                          | 1.5722                              |
| yolov8m-pose | 6.0593                          | 2.1375                              |
| yolov8l-pose | 8.2359                          | 2.9220                              |
| yolov8x-pose | 11.2578                         | 3.6051                              |

</details>