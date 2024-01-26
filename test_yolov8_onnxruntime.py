import time
import requests
from shutil import move
from pathlib import Path
from argparse import ArgumentParser, Namespace

import numpy as np

import torch

# tensorrt must be imported to specify libnvinfer.so.x
import torch_tensorrt
import onnxruntime as ort

from ultralytics import YOLO


def main(opt: Namespace) -> None:
    assert torch.cuda.is_available(), "Can not find Nvidia GPU/TPU devices."

    opt.model: str
    opt.task: str
    match opt.task:
        case "detection":
            task = ""
        case "segmentation":
            task = "-seg"
        case "pose":
            task = "-pose"
        case "obb":
            task = "-obb"
        case "classification":
            task = "-cls"
    model_name = f"{opt.model}{task}"

    torch_model_path = Path(f"./resource/models/torch/{model_name}.pt")
    torch_model_path.parent.mkdir(exist_ok=True, parents=True)
    with open(torch_model_path, "wb") as file_out:
        response = requests.get(
            "/".join(
                [
                    "https://github.com/ultralytics",
                    "assets",
                    "releases",
                    "download",
                    "v8.1.0",
                    f"{model_name}.pt",
                ]
            ),
        )
        file_out.write(response.content)

    print(f"Loading {model_name} from torchvision...")
    model = YOLO(str(torch_model_path))

    # Remove Comment if you want to try
    # try:
    #     scripted_module_path = torch_model_path.with_suffix(".torchscript")
    #
    #     print(f"Scripting {opt.model}...")
    #     model.export(format="torchscript", imgsz=640)
    #
    #     scripted_module = torch.jit.load(
    #         str(scripted_module_path), map_location="cuda:0"
    #     )
    #     scripted_module = scripted_module.half()
    #
    #     tensorrt_model_path = Path(f"./{model_name}.ts")
    #     tensorrt_model_path.parent.mkdir(exist_ok=True, parents=True)
    #     if tensorrt_model_path.exists():
    #         print("TensorRT model already exists. Skip Compiling.")
    #         trt_ts_module = torch.jit.load(str(tensorrt_model_path))
    #     else:
    #         print("Compiling TorchScript into TensorRT...")
    #         trt_ts_module = torch_tensorrt.compile(
    #             scripted_module,
    #             inputs=[
    #                 torch_tensorrt.Input(
    #                     shape=[1, 3, 640, 640],
    #                     dtype=torch.half,
    #                 )
    #             ],
    #             enabled_precisions={torch.half},
    #         )
    #         torch.jit.save(trt_ts_module, tensorrt_model_path)
    # except RuntimeError as err:
    #     print(err)
    #     print(f"Failed to convert {model_name} into tensorrt")

    print(f"Convert {opt.model} in ONNX...")
    onnx_model_path = Path(f"./resource/models/onnx/{model_name}.onnx")
    if onnx_model_path.exists():
        print("ONNX model already exists. Skip Converting.")
    else:
        model.export(
            format="onnx",
            imgsz=640,
            opset=18,
            simplify=True,
            half=True,
            device=0,
        )
        onnx_model_path.parent.mkdir(exist_ok=True, parents=True)
        move(torch_model_path.with_suffix(".onnx"), onnx_model_path)

    onnxruntime_cache_path = Path(
        f"./resource/models/onnxruntime_tensorrt/{model_name}"
    )
    onnxruntime_cache_path.parent.mkdir(exist_ok=True, parents=True)
    session = ort.InferenceSession(
        str(onnx_model_path),
        providers=[
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_fp16_enable": True,
                    "trt_timing_cache_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(onnxruntime_cache_path),
                },
            )
        ],
    )

    num_iter: int = opt.num_iter

    print("PyTorch")
    float32_data = torch.ones([1, 3, 640, 640]).cuda()
    start = time.time()
    for _ in range(num_iter):
        model(float32_data)  # run inference
    end = time.time()
    print(f"\tAverage Inference Time: {(end - start) / num_iter * 1000:.4f}ms")

    print("TensorRT with onnxruntime")
    float16_data_np = np.zeros([1, 3, 640, 640], dtype=np.float16)
    start = time.time()
    for _ in range(num_iter):
        session.run(None, {session.get_inputs()[0].name: float16_data_np})
    end = time.time()
    print(f"\tAverage Inference Time: {(end - start) / num_iter * 1000:.4f}ms")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "yolov8n",
            "yolov8s",
            "yolov8m",
            "yolov8l",
            "yolov8x",
        ],
        help="The model name to test.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="detection",
        choices=[
            "detection",
            "segmentation",
            "pose",
            "obb",
            "classification",
        ],
        help="The task of YOLOv8.",
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=1000,
        help="The number of iterations for testing inference speed.",
    )
    main(parser.parse_args())
