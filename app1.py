from flask import Flask, request, render_template
from ultralytics import YOLO
import base64
import os
import cv2
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Update the main path to your dataset
main_path = r'C:\Users\Pc\Downloads\helmet dataset'
val_images = os.path.join(main_path, 'val', 'images')
output_dir = os.path.join(main_path, 'output_images')
os.makedirs(output_dir, exist_ok=True)

# Define id2class_map and class2color_map
id2class_map = {
    '0': 'with helmet',
    '1': 'without helmet',
    '2': 'rider',
    '3': 'number plate'
}
class2color_map = {
    'with helmet': (0, 255, 128),
    'without helmet': (255, 51, 51),
    'rider': (51, 255, 255),
    'number plate': (224, 102, 255)
}

# Load YOLOv8 model
model = YOLO(r'c:\Users\Pc\runs\detect\helmet_numberplate_model3\weights\best.pt')  # Update with your model path

# In app1.py (or app.py), update the custom_plot function
def custom_plot(results, class2color_map):
    img = results.orig_img.copy()
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0]) if hasattr(box.cls, '__getitem__') else int(box.cls)
        label = results.names[cls_id]
        color = class2color_map.get(label, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label} {box.conf[0].item():.2f}" if hasattr(box.conf, '__getitem__') else f"{label} {box.conf.item():.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

# Convert image to base64 for web display
def img_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

# Route for home page
@app.route('/', methods=['GET', 'POST'])
def index():
    val_img_files = [f for f in os.listdir(val_images) if f.endswith(('.jpg', '.png'))][:5]
    processed_images = []

    if request.method == 'POST':
        # Handle uploaded image
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            img = Image.open(file.stream)
            img = np.array(img.convert('RGB'))
            results = model.predict(img, imgsz=640, conf=0.5)[0]
            annotated_img = custom_plot(results, class2color_map)
            output_path = os.path.join(output_dir, f"pred_uploaded_{file.filename}")
            cv2.imwrite(output_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
            processed_images.append({'name': file.filename, 'img_base64': img_to_base64(annotated_img)})

        # Handle selected validation images
        selected_val_images = request.form.getlist('val_images')
        for img_name in selected_val_images:
            if img_name in val_img_files:
                img_path = os.path.join(val_images, img_name)
                results = model.predict(img_path, imgsz=640, conf=0.5)[0]
                annotated_img = custom_plot(results, class2color_map)
                output_path = os.path.join(output_dir, f"pred_val_{img_name}")
                cv2.imwrite(output_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
                processed_images.append({'name': img_name, 'img_base64': img_to_base64(annotated_img)})

    return render_template('app.html', val_images=val_img_files, processed_images=processed_images)

if __name__ == '__main__':
    app.run(debug=True)