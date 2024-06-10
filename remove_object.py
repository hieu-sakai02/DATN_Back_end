import torch
import numpy as np
from pathlib import Path
from lama_inpaint import inpaint_img_with_lama
from utils.utils import load_img_to_array, save_array_to_img
import cv2

input_img_path = "./uploaded_image.jpg"
mask_img_path = "./mask_image.jpg"
output_dir = "./"
lama_config_path = "./lama/configs/prediction/default.yaml"
lama_ckpt_path = "./big-lama"

def remove_object(device):
    img = load_img_to_array(input_img_path)
    mask = load_img_to_array(mask_img_path, convert_to_grayscale=True)
    
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_resized = (mask_resized > 128).astype(np.uint8) * 255

    img_stem = Path(input_img_path).stem
    out_dir = Path(output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    
    img_inpainted_p = f"./inpainted_image.jpg"
    print(img_inpainted_p)
    img_inpainted = inpaint_img_with_lama(
        img, mask_resized, lama_config_path, lama_ckpt_path, device=device)
    save_array_to_img(img_inpainted, img_inpainted_p)

    return 'inpainted_image.jpg'