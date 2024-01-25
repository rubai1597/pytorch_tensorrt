import time
import copy
from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
from torchvision.models.resnet import (
    resnet18,
    ResNet18_Weights,
    resnet34,
    ResNet34_Weights,
    resnet50,
    ResNet50_Weights,
    resnet101,
    ResNet101_Weights,
    resnet152,
    ResNet152_Weights,
)

import torch_tensorrt


def main(opt: Namespace) -> None:
    assert torch.cuda.is_available(), "Can not find Nvidia GPU/TPU devices."

    opt.model: str
    print(f"Loading {opt.model} from torchvision...")
    match opt.model:
        case "resnet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        case "resnet34":
            model = resnet34(weights=ResNet34_Weights.DEFAULT)
        case "resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        case "resnet101":
            model = resnet101(weights=ResNet101_Weights.DEFAULT)
        case "resnet152":
            model = resnet152(weights=ResNet152_Weights.DEFAULT)
    model = model.eval().cuda()

    print(f"Scripting {opt.model}...")
    model_fp16 = copy.deepcopy(model)
    model_fp16 = model_fp16.half()
    scripted_module = torch.jit.script(model_fp16)

    tensorrt_model_path = Path(f"{opt.model}.ts")
    if tensorrt_model_path.exists():
        print("TensorRT model already exists. Skip Compiling.")
        trt_ts_module = torch.jit.load(tensorrt_model_path)
    else:
        print("Compiling TorchScript into TensorRT...")
        trt_ts_module = torch_tensorrt.compile(
            scripted_module,
            inputs=[
                torch_tensorrt.Input(  # Specify input object with shape and dtype
                    min_shape=[1, 3, 112, 112],
                    opt_shape=[1, 3, 224, 224],
                    max_shape=[1, 3, 448, 448],
                    # For static size shape=[1, 3, 224, 224]
                    dtype=torch.half,
                )  # Datatype of input tensor. Allowed options torch.(float|half|int32|bool)
            ],
            enabled_precisions={torch.half},  # Run with FP16
        )
        torch.jit.save(trt_ts_module, tensorrt_model_path)

    num_iter = 1000

    print("PyTorch")
    float32_data = torch.ones([1, 3, 224, 224]).cuda()
    start = time.time()
    for _ in range(num_iter):
        model(float32_data)  # run inference
    end = time.time()
    print(f"\tAverage Inference Time: {(end - start) / num_iter * 1000:.4f}ms")

    print("TensorRT with TorchScript")
    float16_data = torch.ones([1, 3, 224, 224], dtype=torch.half).cuda()
    start = time.time()
    for _ in range(num_iter):
        trt_ts_module(float16_data)  # run inference
    end = time.time()
    print(f"\tAverage Inference Time: {(end - start) / num_iter * 1000:.4f}ms")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        ],
        help="The model name to test.",
    )
    main(parser.parse_args())
