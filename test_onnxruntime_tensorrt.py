import time
import copy
from pathlib import Path
from argparse import ArgumentParser, Namespace

import numpy as np
from PIL import Image

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

# tensorrt must be imported to specify libnvinfer.so.x
import tensorrt
import onnxruntime as ort


def main(opt: Namespace) -> None:
    assert torch.cuda.is_available(), "Can not find Nvidia GPU/TPU devices."

    opt.model: str
    print(f"Loading {opt.model} from torchvision...")
    match opt.model:
        case "resnet18":
            module = resnet18
            weights = ResNet18_Weights.DEFAULT
        case "resnet34":
            module = resnet34
            weights = ResNet34_Weights.DEFAULT
        case "resnet50":
            module = resnet50
            weights = ResNet50_Weights.DEFAULT
        case "resnet101":
            module = resnet101
            weights = ResNet101_Weights.DEFAULT
        case "resnet152":
            module = resnet152
            weights = ResNet152_Weights.DEFAULT

    model = module(weights=weights)
    model = model.eval().cuda()

    print(f"Convert {opt.model} in ONNX...")
    onnx_model_path = Path(f"./resource/models/onnx/{opt.model}.onnx")
    if onnx_model_path.exists():
        print("ONNX model already exists. Skip Converting.")
    else:
        model_fp16 = copy.deepcopy(model)
        model_fp16 = model_fp16.half()

        onnx_model_path.parent.mkdir(exist_ok=True, parents=True)
        torch.onnx.export(
            model_fp16,
            torch.zeros((1, 3, 224, 224), dtype=torch.half, device="cuda:0"),
            str(onnx_model_path),
            opset_version=18,
        )

    onnxruntime_cache_path = Path(f"./resource/models/onnxruntime_tensorrt/{opt.model}")
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
    float32_data = torch.ones([1, 3, 224, 224]).cuda()
    start = time.time()
    for _ in range(num_iter):
        model(float32_data)  # run inference
    end = time.time()
    print(f"\tAverage Inference Time: {(end - start) / num_iter * 1000:.4f}ms")

    print("TensorRT with onnxruntime")
    float16_data_np = np.zeros([1, 3, 224, 224], dtype=np.float16)
    start = time.time()
    for _ in range(num_iter):
        session.run(None, {session.get_inputs()[0].name: float16_data_np})
    end = time.time()
    print(f"\tAverage Inference Time: {(end - start) / num_iter * 1000:.4f}ms")

    print("Compare predictions between PyTorch and TensorRT.")
    img_path: str = opt.test_img
    if img_path.startswith("https://") or img_path.startswith("http://"):
        import requests
        from io import BytesIO

        try:
            response = requests.get(img_path)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise IOError(f"Image URL({img_path}) is not valid") from e
    else:
        image = Image.open(img_path).convert("RGB")

    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    batch = preprocess(image)
    batch = batch.unsqueeze(0).cuda()

    topk = 5

    prediction_torch = model(batch).squeeze(0).softmax(0)
    class_ids = torch.argsort(-prediction_torch)[:topk]
    scores = prediction_torch[class_ids]
    print("PyTorch Predictions:")
    for class_id, score in zip(class_ids, scores):
        print(f"\t{categories[class_id]:20s}: {score:10.4f}")

    onnx_outputs = session.run(
        None, {session.get_inputs()[0].name: batch.cpu().numpy().astype(np.float16)}
    )[0]
    prediction_tensorrt = (
        torch.from_numpy(onnx_outputs).squeeze(0).to(torch.float32).softmax(0)
    )
    class_ids = torch.argsort(-prediction_tensorrt)[:topk]
    scores = prediction_tensorrt[class_ids]
    print("TensorRT Predictions:")
    for class_id, score in zip(class_ids, scores):
        print(f"\t{categories[class_id]:20s}: {score:10.4f}")


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
    parser.add_argument(
        "--num_iter",
        type=int,
        default=1000,
        help="The number of iterations for testing inference speed.",
    )
    parser.add_argument(
        "--test_img",
        type=str,
        default="golden_retrieval.jpg",
        help="The image path/url to test.",
    )
    main(parser.parse_args())
