
# Road Compilance AI 
**AI-Based Helmet & Number Plate Detection System using YOLOv8n**

 Road Compilance AI is a real-time computer vision system built using YOLOv8n to monitor and detect helmet usage and number plates of motorcycle riders. Designed for traffic surveillance and road safety enforcement, the system processes images and videos to identify helmet violations and extract license plate data.

---

##  Features
- **Helmet Detection**: Accurately detects whether motorcycle riders are wearing helmets.
- **Number Plate Recognition**: Identifies and localizes vehicle number plates for further processing.
- **Dual YOLO Support**: Compatible with  YOLOv8n models for flexible deployment.
- **Real-Time Processing**: Handles images, video files, and live camera feeds with low latency.
- **Custom Training**: Utilizes a dataset tailored for helmet and number plate detection to improve accuracy.
- **Result Visualization**: Outputs annotated images with bounding boxes and class labels for easy review.
- **Easy Integration**: Modular codebase designed for quick adaptation to new requirements or environments.
- **Edge & Cloud Ready**: Scalable for use on local machines, edge devices, or cloud servers.
- **User-Friendly Configuration**: Simple setup for dataset paths, model weights, and detection thresholds.

---

## Project Structure

```
Road compilace Ai/
├── data/
│   ├── images/               # Training/testing images
│   ├── labels/               # YOLO-format labels
│
├── models/                   # Trained weights
│   └── best.pt               # Best trained YOLO model
│
├── output_images/            # Annotated result images
│
├── detect.py                 # Inference script
├── train.py                  # Training script
├── data.yaml                 # Class names and dataset paths
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```


##  Configuration

- **Dataset**: Add your YOLO-annotated images to `data/images` and corresponding label files to `data/labels`.
- **Class Names**: Edit `data.yaml` to include your detection classes:
  ```yaml
  names: ['with helmet', 'without helmet', 'Rider', 'number_plate']
  ```
- **Model Weights**: Save your trained model file (e.g., `best.pt`) in the `models/` folder.
- **Detection Thresholds**: Modify confidence or IoU thresholds in `detect.py` to fine-tune detection sensitivity.

---

##  Training the Model

```bash
# Train with YOLOv8
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```

---

##  Running Inference

```bash
# Run inference with YOLOv8
yolo task=detect mode=predict model=models/best.pt source=data/images/ conf=0.4
```

- Annotated images will be saved in the `runs/detect/` directory.

---

##  Customization

- **Detection Classes**: Update the `names` field in `data.yaml` to match your specific detection targets.
- **Video Input Source**: In inference commands, set the `source` parameter to a webcam index (e.g., `0`) or provide a path to a video file for custom input.
- **Output Directory**: Adjust the output directory path in `detect.py` to control where annotated results are saved.

---

##  Use Cases

- **Traffic Violation Monitoring**: Automatically detects and logs helmet and number plate violations for improved road safety.
- **Smart City Surveillance**: Integrates with urban monitoring systems to enhance public safety and compliance.
- **Law Enforcement & Fine Generation**: Assists authorities in identifying offenders and streamlining the process of issuing fines.
- **Helmet Compliance Reporting**: Generates reports and analytics on helmet usage trends for policy-making and awareness campaigns.
---


##  Acknowledgments
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) – for the core object detection framework.
- [Kaggle Helmet Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/helmet-detection) – for training and evaluation data.
- [OpenCV](https://opencv.org/) – for image and video processing.
---

> “The safest helmet is the one you wear — and now, Road Compilance Ai helps ensure it’s there.”  
> — Road Compilance Ai

