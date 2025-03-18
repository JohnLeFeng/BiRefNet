import os
import sys

import numpy as np
import openvino as ov

from PIL import Image
from pathlib import Path
from image_proc import refine_foreground

# For postprocessing
import openvino.runtime.opset12 as ops
from openvino.runtime import Output
from openvino.runtime.utils.decorators import custom_preprocess_function

@custom_preprocess_function
def custom_sigmoid(output: Output):
    # Custom nodes can be inserted as Post-processing steps
    return ops.sigmoid(output)

def main():
    ov_model_path = Path("./ov_model/FP16/BiRefNet-general-bb_swin_v1_tiny.xml")
    image_path = Path("test_image/gettyimages-1229892983-square.jpg")

    image = Image.open(image_path)
    image = image.convert("RGB") if image.mode != "RGB" else image

    core = ov.Core()
    core.set_property(properties={'CACHE_DIR': './cache', "PERFORMANCE_HINT": "LATENCY"}, device_name='GPU')

    if not os.path.exists(ov_model_path):
        onnx_model_path = Path("onnx_model/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx")

        ov_model = core.read_model(model=onnx_model_path)

        input_image = np.transpose(image, (2, 0, 1))
        input_tensor = np.expand_dims(input_image, 0)

        prep = ov.preprocess.PrePostProcessor(ov_model)
        prep.input(0).tensor().set_layout(ov.Layout("NCHW"))
        prep.input(0).preprocess().scale([255, 255, 255])
        prep.input(0).preprocess().mean([0.485, 0.456, 0.406]).scale([0.229, 0.224, 0.225])
        prep.output(0).postprocess().custom(custom_sigmoid)

        ov_model = prep.build()
        ov.save_model(ov_model, ov_model_path)
        compiled_model = core.compile_model(ov_model, "CPU")
    else:
        compiled_model = core.compile_model(ov_model_path, "CPU")
    
    pred = compiled_model(np.expand_dims(np.transpose(np.array(image.resize((1024, 1024))), (2, 0, 1)), 0))[0]

    pred_pil = Image.fromarray((pred[0][0] * 255).astype(np.uint8), mode='L')

    image_masked = refine_foreground(image, pred_pil)
    image_masked.putalpha(pred_pil.resize(image.size))
    image_masked.save("ov_result.png")

if __name__ == '__main__':
    sys.exit(main())