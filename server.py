from flask import Flask, request, jsonify
import generate_mask as gm
import generate_image as gi
import remove_object as ro
from segment_anything import sam_model_registry
import os
import torch
from diffusers import StableDiffusionInpaintPipeline
import base64

app = Flask(__name__)

HOME = os.getcwd()
WEIGHTS_DIR = os.path.join(HOME, "weights")
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "sam_vit_h_4b8939.pth")
print("start download model")
MODEL_TYPE = "vit_h"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
print("finish download model")

MODEL_PATH = os.path.join(WEIGHTS_DIR, "sd-v1-5-inpainting")

print(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    ).to(device=DEVICE)
    pipe.save_pretrained(MODEL_PATH)
else:
    pipe = StableDiffusionInpaintPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device=DEVICE)
    

# Generate Mask API Route
@app.route("/generate-mask", methods=["POST"])
def generate_mask_route():
    uploaded_image = request.files['image']
    image_path = "uploaded_image.jpg"
    uploaded_image.save(image_path)

    print("start generate mask")

    generated_mask_images = gm.generate_mask(image_path, sam)
    
    return jsonify(generated_mask_images)

@app.route("/save-image", methods=["POST"])
def save_image_route():
    uploaded_image = request.files['image']
    image_path = "uploaded_image.jpg"
    uploaded_image.save(image_path)
    return "Save image complete"

@app.route("/generate-image", methods=["POST"])
def generate_image_route():
    data = request.json
    prompt = data.get('prompt')
    
    mask_image_base64 = data.get('mask_image')
        
    if not mask_image_base64:
        return jsonify({"status": "error", "message": "No mask image provided"}), 400

    try:
        if ',' in mask_image_base64:
            mask_image_data = base64.b64decode(mask_image_base64.split(',')[1])
        else:
            mask_image_data = base64.b64decode(mask_image_base64)

        mask_image_path = "mask_image.jpg"
        # Save the image to the server
        with open(mask_image_path, "wb") as f:
            f.write(mask_image_data)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    try:
        # Generate the image based on the prompt and mask image
        generated_image_path = gi.generate_image(prompt, pipe)
        
        # Read the generated image and encode it in base64
        with open(generated_image_path, "rb") as image_file:
            generated_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        return jsonify({"status": "success", "generated_image": generated_image_base64, "prompt": prompt})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route("/remove-object", methods=["POST"])
def remove_object():
    data = request.json
    
    mask_image_base64 = data.get('mask_image')
        
    if not mask_image_base64:
        return jsonify({"status": "error", "message": "No mask image provided"}), 400

    try:
        if ',' in mask_image_base64:
            mask_image_data = base64.b64decode(mask_image_base64.split(',')[1])
        else:
            mask_image_data = base64.b64decode(mask_image_base64)

        mask_image_path = "mask_image.jpg"
        # Save the image to the server
        with open(mask_image_path, "wb") as f:
            f.write(mask_image_data)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
    try:
        # Generate the image based on the prompt and mask image
        generated_image_path = ro.remove_object(DEVICE)
        
        # Read the generated image and encode it in base64
        with open(generated_image_path, "rb") as image_file:
            generated_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        return jsonify({"status": "success", "generated_image": generated_image_base64})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)