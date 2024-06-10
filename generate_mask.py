import cv2
from PIL import Image
import supervision as sv 
from segment_anything import SamAutomaticMaskGenerator
import numpy as np
import base64

def generate_mask(IMAGE_PATH, sam):
    image_path = IMAGE_PATH
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_bgr = cv2.resize(image_bgr, (512, 512), interpolation=None)
    image_rgb = cv2.resize(image_rgb, (512, 512), interpolation=None)

    mask_generator = SamAutomaticMaskGenerator(sam)
    result = mask_generator.generate(image_rgb)

    base64_masks = []
    for segment in sorted(result, key=lambda x: x['area'], reverse=True)[:10]:
        segmentation = segment['segmentation']
        # Convert boolean array to binary image
        binary_mask = np.uint8(segmentation) * 255
        # Encode binary image as base64 string
        _, buffer = cv2.imencode('.png', binary_mask)
        base64_mask = base64.b64encode(buffer).decode('utf-8')
        base64_masks.append(base64_mask)    
        
    return base64_masks