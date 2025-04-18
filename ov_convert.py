# Imports
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from IPython.display import display
from utils import check_state_dict
import os
from models.birefnet import BiRefNet
from image_proc import refine_foreground
import openvino as ov
from pathlib import Path

import openvino.runtime.opset12 as ops
from openvino.runtime import Output
from openvino.runtime.utils.decorators import custom_preprocess_function
import argparse

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    # fmt: off
    args.add_argument('-h', '--help', action = 'help',
                      help='Show this help message and exit.')
    args.add_argument('-ih', '--input_height', type = int, default = 224,
                      help='Optional. Height of input.')
    args.add_argument('-iw', '--input_width', type = int, default = 224,
                      help='Optional. Width of input.')

    return parser.parse_args()

args = parse_args()

WIDTH = args.input_width
HEIGHT = args.input_height

@custom_preprocess_function
def custom_sigmoid(output: Output):
    # Custom nodes can be inserted as Post-processing steps
    return ops.sigmoid(output)

def add_output_name(ov_model, output_names):
    _outputs = ov_model.outputs
    for _idx, _output in enumerate(_outputs):
        _output.set_names({output_names[_idx]})
    return ov_model

if __name__ == "__main__":
    # 
    # Thanks Chen, Tianmeng (tianmeng.chen@intel.com) to use torch to convert to IR.
    # 
    birefnet = BiRefNet(bb_pretrained=False)
    # Please download model from:https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-bb_swin_v1_tiny-epoch_232.pth
    state_dict = torch.load('torch_model/BiRefNet-general-bb_swin_v1_tiny-epoch_232.pth', map_location='cpu', weights_only=True)
    # birefnet = BiRefNet(bb_pretrained=False, bb='swin_v1_l')
    # Please download model from:https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-resolution_512x512-fp16-epoch_216.pth
    # state_dict = torch.load('torch_model/BiRefNet-general-resolution_512x512-fp16-epoch_216.pth', map_location='cpu', weights_only=True)
    state_dict = check_state_dict(state_dict)
    birefnet.load_state_dict(state_dict)

    device = 'cpu'

    torch.set_float32_matmul_precision(['high', 'highest'][0])

    birefnet.to(device)
    birefnet.eval()

    # Input Data
    transform_image = transforms.Compose([
        transforms.Resize((WIDTH, HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # image_path = Path("test_image/gettyimages-1229892983-square.jpg")
    image_path = Path("test_image/Joelan.jpg")
    image = Image.open(image_path)
    image = image.convert("RGB") if image.mode != "RGB" else image

    # Prediction
    with torch.no_grad():
        preds = birefnet(transform_image(image).unsqueeze(0).to(device))[-1].sigmoid().cpu()
    pred = preds[0].squeeze()

    # # Show Results
    pred_pil = transforms.ToPILImage()(pred)
    pred_pil.resize(image.size).save("torch_mask/torch_Joelan_mask_result_" + str(WIDTH) + "x" + str(HEIGHT) + ".jpg")

    # convert
    core = ov.Core()
    ov_model_path = Path("ov_model/FP16/BiRefNet-general-bb_swin_v1_tiny_from_torch_" + str(WIDTH) + "x" + str(HEIGHT) + ".xml")
    # ov_model_path = Path("ov_model/FP16/BiRefNet-general-resolution_512x512_from_torch_" + str(WIDTH) + "x" + str(HEIGHT) + ".xml")
    if not os.path.exists(ov_model_path):
        input_images = transform_image(image).unsqueeze(0).to(device)
        example_input = input_images
        ov_model = ov.convert_model(birefnet, example_input=example_input, input=[1, 3, HEIGHT, WIDTH])

        prep = ov.preprocess.PrePostProcessor(ov_model)
        prep.input(0).tensor().set_layout(ov.Layout("NCHW"))
        prep.input(0).preprocess().scale([255, 255, 255])
        prep.input(0).preprocess().mean([0.485, 0.456, 0.406]).scale([0.229, 0.224, 0.225])
        prep.output(0).postprocess().custom(custom_sigmoid)

        ov_model = prep.build()
        ov_model = add_output_name(ov_model, ['output'])
        ov.save_model(ov_model, ov_model_path)
        compiled_model = core.compile_model(ov_model, "GPU")
    else:
        compiled_model = core.compile_model(ov_model_path, "GPU")
    #Infer
    pred = compiled_model(np.expand_dims(np.transpose(np.array(image.resize((WIDTH, HEIGHT))), (2, 0, 1)), 0))[0]
    pred_pil = Image.fromarray((pred[0][0] * 255).astype(np.uint8), mode='L')
    pred_pil.resize(image.size).save("ov_mask/ov_Joelan_mask_result_" + str(WIDTH) + "x" + str(HEIGHT) + ".jpg")

    image_masked = refine_foreground(image, pred_pil)
    image_masked.putalpha(pred_pil.resize(image.size))
    image_masked.save("output_image\ov_Joelan_result_" + str(WIDTH) + "x" + str(HEIGHT) + ".png")
