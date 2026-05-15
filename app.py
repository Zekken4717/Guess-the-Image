import base64
import io
import os
import random
from flask import Flask, render_template, jsonify, request
import numpy as np
from PIL import Image

app = Flask(__name__)
app = app # Required for Vercel routing discovery

IMAGE_FILES = [
    "images/apples.jpeg", 
    "images/raspberry.jpeg", 
    "images/strawberries.png", 
    "images/watermelon.jpg"
]
FRUIT_NAMES = ["apples", "raspberry", "strawberries", "watermelon"]

def compress_and_render_b64(image_path, n, make_grayscale=False):
    """Processes SVD and renders the final image to a tiny Base64 string."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Missing file: {image_path}")

    img_obj = Image.open(image_path).convert('RGB')
    img_array = np.array(img_obj)
    height, width, _ = img_array.shape
    
    # Core mathematical SVD processing rank threshold cap
    max_rank = min(height, width)
    n_components = min(max(1, int(n)), max_rank)
    
    reconstructed_channels = []
    for channel in range(3):
        channel_matrix = img_array[:, :, channel].astype(float)
        U, s, V = np.linalg.svd(channel_matrix, full_matrices=False)
        
        # Keep only the targeted n components
        U_n = U[:, :n_components]
        s_n = s[:n_components]
        V_n = V[:n_components, :]
        
        recon_channel = U_n @ np.diag(s_n) @ V_n
        reconstructed_channels.append(recon_channel)
        
    # Reassemble matrix structure
    recon_img = np.stack(reconstructed_channels, axis=2)
    
    # Normalize numerical boundaries securely between 0 and 255
    recon_img = np.clip(recon_img, 0, None)
    max_val = recon_img.max() if recon_img.max() > 0 else 1
    rescaled = (255.0 / max_val * (recon_img - recon_img.min())).astype(np.uint8)
    
    final_image = Image.fromarray(rescaled)
    
    # Apply standard gray weight scaling luminance formula if active
    if make_grayscale:
        final_image = final_image.convert('L')
        
    # Compress matrix down to a lightweight PNG format byte stream
    buf = io.BytesIO()
    final_image.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route("/")
def index():
    try:
        # Pick the starting image state parameters
        rand_int = random.randint(0, len(IMAGE_FILES) - 1)
        
        # Cache round selections within hidden DOM tracking fields
        initial_img = IMAGE_FILES[rand_int]
        correct_name = FRUIT_NAMES[rand_int]
        
        # Render baseline state image safely under Vercel payload limits
        img_b64 = compress_and_render_b64(initial_img, n=1, make_grayscale=False)
        
        return render_template(
            'index.html',
            correct=correct_name,
            initial_image_b64=img_b64,
            current_image_path=initial_img,
            names_list_json=jsonify(FRUIT_NAMES).get_data(as_text=True)
        )
    except Exception as e:
        return f"<h3>Backend initialization failed: {str(e)}</h3>", 500

@app.route("/api/render-image", channels=['GET', 'POST'])
@app.route("/api/render-image")
def render_image_api():
    """Handles slider, inputs, and toggles by sending only optimized Base64."""
    try:
        img_path = request.args.get('path', IMAGE_FILES[0])
        n = int(request.args.get('n', 1))
        grayscale_param = request.args.get('grayscale', 'false').lower() == 'true'
        
        img_b64 = compress_and_render_b64(img_path, n=n, make_grayscale=grayscale_param)
        return jsonify({"img_b64": img_b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/new-round")
def new_round_api():
    """Selects a new fruit target configuration instantly."""
    try:
        rand_int = random.randint(0, len(IMAGE_FILES) - 1)
        chosen_path = IMAGE_FILES[rand_int]
        correct_name = FRUIT_NAMES[rand_int]
        
        img_b64 = compress_and_render_b64(chosen_path, n=1, make_grayscale=False)
        return jsonify({
            "correct": correct_name,
            "img_path": chosen_path,
            "img_b64": img_b64
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
