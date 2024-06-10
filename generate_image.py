from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

def generate_image(prompt, pipe):
    img_path = 'uploaded_image.jpg' 
    init_image = Image.open(img_path)

    mask_path = 'mask_image.jpg'
    mask_image = Image.open(mask_path)

    # Resize the image if needed
    init_image = init_image.resize((512, 512))
    mask_image = mask_image.resize((512, 512))

    # Define your prompt
    prompt = prompt
    
    print("prompt: " + prompt)

    print("start generate image")
    # Perform inpainting
    image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

    # Save the inpainted image
    file_path = 'inpainted_image.jpg'  # Change this to the path where you want to save the image
    image.save(file_path)

    return file_path