import os
import sys
import nncf
import torch
import logging

from PIL import Image
from pathlib import Path
from argparse import ArgumentParser, SUPPRESS
from torch.utils.data import Subset
from torchvision import datasets
from torchvision import transforms

import openvino as ov
import numpy as np

from image_proc import refine_foreground


logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, 
                      help='Show this help message and exit.')
    args.add_argument('-ss', '--subset_size', type = int, default = 100,
                      help='Optional. Number for subset size for quantization.')
    args.add_argument('-ih', '--height', type=int, required=True,
                        help="Required. Height of the input image")
    args.add_argument('-iw', '--width', type=int, required=True,
                        help="Required. Width of the input image")

    return parser.parse_args()

def transform_fn(data_item):
    images, _ = data_item
    return images

FP8_FORMAT = {
    "FP8_E4M3": nncf.QuantizationMode.FP8_E4M3,
    "FP8_E5M2": nncf.QuantizationMode.FP8_E5M2,
}

args = parse_args()

SUBSET_SIZE = args.subset_size

HEIGHT = args.height
WIDTH = args.width

# DATASET: "schirrmacher/humans"
#     https://huggingface.co/datasets/schirrmacher/humans
# https://drive.google.com/drive/folders/1GRzFDE0VbIEIp_XuR44SYekMqqJa-U8Y

SMOOTH_QUANT_ALPHA = 0.15

FP16_OV_MODEL_PATH = Path("ov_model/FP16/BiRefNet-general-bb_swin_v1_tiny_from_torch_" + str(WIDTH) + "x" + str(HEIGHT) + ".xml")
INT8_OV_MODEL_PATH = Path("ov_model/INT8/BiRefNet-general-bb_swin_v1_tiny_from_torch_" + str(WIDTH) + "x" + str(HEIGHT) + ".xml")
FP8E4M3_OV_MODEL_PATH = Path("ov_model/FP8E4M3/BiRefNet-general-bb_swin_v1_tiny_from_torch_" + str(WIDTH) + "x" + str(HEIGHT) + ".xml")

def main():
    core = ov.Core()

    ov_model = core.read_model(FP16_OV_MODEL_PATH)

    dataset = datasets.ImageFolder(
        root="./Dataset/humans/dataset/validation/",
        transform=transforms.Compose(
            [
                transforms.Resize([HEIGHT, WIDTH]),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 255)
            ]
        ),
    )

    dataset = Subset(dataset, [i for i in range(len(dataset)) if dataset[i][1] == 2])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    calibration_dataset = nncf.Dataset(data_loader, transform_fn)

    if not INT8_OV_MODEL_PATH.exists():
        log.info ("Quantizing INT8 model is exist.")
        ov_quantized_INT8_model  = nncf.quantize(
            model=ov_model,
            subset_size=SUBSET_SIZE,
            preset=nncf.QuantizationPreset.PERFORMANCE,
            calibration_dataset=calibration_dataset,
            model_type=nncf.ModelType.TRANSFORMER,
            advanced_parameters=nncf.AdvancedQuantizationParameters(
                disable_bias_correction=True,
                smooth_quant_alphas=nncf.AdvancedSmoothQuantParameters(matmul=SMOOTH_QUANT_ALPHA),
            )
        )
        ov.save_model(ov_quantized_INT8_model , INT8_OV_MODEL_PATH)
        log.info ("Saved Quantized INT8 model.")
    else:
        log.info ("Quantized INT8 model is exist.")

    if not FP8E4M3_OV_MODEL_PATH.exists():
        log.info ("Quantizing FP8_E4M3 model is exist.")
        ov_quantized_FP8_E4M3_model  = nncf.quantize(
            model=ov_model,
            subset_size=SUBSET_SIZE,
            preset=nncf.QuantizationPreset.PERFORMANCE,
            mode=FP8_FORMAT["FP8_E4M3"],
            calibration_dataset=calibration_dataset,
            model_type=nncf.ModelType.TRANSFORMER,
            advanced_parameters=nncf.AdvancedQuantizationParameters(
                disable_bias_correction=True,
                smooth_quant_alphas=nncf.AdvancedSmoothQuantParameters(matmul=SMOOTH_QUANT_ALPHA),
            )
        )
        ov.save_model(ov_quantized_FP8_E4M3_model , FP8E4M3_OV_MODEL_PATH)
        log.info ("Saved Quantized FP8_E4M3 model.")
    else:
        log.info ("Quantized FP8_E4M3 model is exist.")

    image_path = Path("test_image/Joelan.jpg")
    image = Image.open(image_path)
    image = image.convert("RGB") if image.mode != "RGB" else image

    compiled_model_fp8e4m3 = core.compile_model(FP8E4M3_OV_MODEL_PATH, "CPU")
    log.info ("Loaded FP8_E4M3 model is exist.")
    compiled_model_int8 = core.compile_model(INT8_OV_MODEL_PATH, "CPU")
    log.info ("Loaded INT8 model is exist.")

    #Infer
    pred = compiled_model_fp8e4m3(np.expand_dims(np.transpose(np.array(image.resize((WIDTH, HEIGHT))), (2, 0, 1)), 0))[0]
    pred_pil = Image.fromarray((pred[0][0] * 255).astype(np.uint8), mode='L')
    pred_pil.resize(image.size).save("ov_mask/ov_Joelan_mask_result_" + str(WIDTH) + "x" + str(HEIGHT) + "_fp8e4m3.jpg")

    image_masked = refine_foreground(image, pred_pil)
    image_masked.putalpha(pred_pil.resize(image.size))
    image_masked.save("output_image\ov_Joelan_result_" + str(WIDTH) + "x" + str(HEIGHT) + "_fp8e4m3.png")
    log.info ("Saved FP8_E4M3 result.")

    #Infer
    pred = compiled_model_int8(np.expand_dims(np.transpose(np.array(image.resize((WIDTH, HEIGHT))), (2, 0, 1)), 0))[0]
    pred_pil = Image.fromarray((pred[0][0] * 255).astype(np.uint8), mode='L')
    pred_pil.resize(image.size).save("ov_mask/ov_Joelan_mask_result_" + str(WIDTH) + "x" + str(HEIGHT) + "_int8.jpg")

    image_masked = refine_foreground(image, pred_pil)
    image_masked.putalpha(pred_pil.resize(image.size))
    image_masked.save("output_image\ov_Joelan_result_" + str(WIDTH) + "x" + str(HEIGHT) + "_int8.png")
    log.info ("Saved INT8 result.")

if __name__ == "__main__":
    sys.exit(main())