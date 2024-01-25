# <div align=center>Installation</div>
## 1. Install TensorRT from the Python Package.
This step referred to [nvidia tensorrt quide](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#installing-pip) and [pytorch tensorrt guide](https://pytorch.org/TensorRT/getting_started/installation.html#installation).

### 1) create Anaconda environment
```bash
conda create -n torch_tensorrt python=3.10 -y
conda activate torch_tensorrt
```

### 2) Install tensorrt

```bash
pip install packaging==23.2
pip install tensorrt==8.6.1
pip install tensorrt_libs==8.6.1
pip install tensorrt_bindings==8.6.1
```

### 3) Install torch-tensorrt and torchvision
pytorch(2.0.1) will be installed together when torch-tensorrt is installed. All you need need to od is install the torchvision, manually.

```bash
pip install torch-tensorrt==1.4.0
pip install torchvision==0.15.2
```

## 2. Test PyTorch Models Converted to TensorRT.
```bash
python main.py --model resnet50
```
