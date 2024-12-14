# YOLOv5-Based Object Detection for Real-Time Object Recognition

## Project Overview
This project leverages YOLOv5, a deep learning-based object detection model, to analyze video data and detect objects in real-time. Object detection technology enables the identification and tracking of specific objects (e.g., blankets, furniture) within video frames efficiently and accurately.

### Key Steps:
1. **Video Data Collection:** Use videos such as `TEST1.mp4`, `TEST2.mp4`, etc.
2. **YOLOv5 Model Training:** Enhance detection accuracy using a pre-trained YOLOv5 model.
![image](https://github.com/user-attachments/assets/5f59f655-68ee-440d-8304-a6b2bd6436bf)


4. **Object Detection:** Analyze videos to store and visualize detected object information.
5. **Webcam Detection:** Detect objects in real-time from live video feeds.

## Background Information
### Necessity of Robot Vacuums
Robot vacuums have become essential household appliances, reducing the burden of domestic chores. Efficient obstacle avoidance and focused cleaning on specific areas are critical features.

### Role of Object Detection
Conventional robot vacuums rely on basic sensors, which limit their ability to recognize obstacles effectively or navigate complex environments. YOLOv5's object detection capabilities can enable adaptive behaviors in robot vacuums.

### YOLOv5 Features
YOLOv5 is a lightweight deep learning model suitable for real-time object detection. It performs efficiently on both GPUs and CPUs, making it adaptable for hardware-limited platforms like robot vacuums.

## Project Objectives
1. **Obstacle and Object Detection:** Design a system where a robot vacuum can detect obstacles and specific objects (e.g., chairs, tables, toys).
2. **Dynamic Navigation:** Enable adaptive cleaning paths and focused cleaning around detected objects.

## Project Phases
1. **Data Collection:** Capture images and videos in diverse environments where robot vacuums operate.
2. **Data Labeling:** Use tools like DarkLabel for annotating objects in the dataset.
3. **YOLOv5 Model Training:** Train the YOLOv5 model using the labeled dataset.
4. **Object Detection Implementation:** Deploy the trained model to detect objects in videos and real-time streams.

## Project Strengths
1. **Real-Time Obstacle Avoidance:** YOLOv5 enables efficient real-time obstacle detection and path optimization.
2. **Targeted Object Detection:** Focused cleaning or avoidance around specific objects (e.g., toys, cables).
3. **Broad Applicability:** Applicable in households, offices, factories, and other environments to enhance robot vacuum performance.

## Project Significance
1. **Advancing Smart Homes:** The project aligns with IoT and AI integration trends in smart home environments.
2. **Improving Efficiency:** Enhances performance compared to sensor-based robot vacuums.
3. **Industry Potential:** The approach can extend beyond robot vacuums to warehouse and logistics robots.

## Challenges
1. **Data Scarcity:** Limited datasets with diverse environments and objects may restrict model performance.
2. **Model Optimization:** While YOLOv5 is efficient, further optimization is necessary for hardware-constrained platforms like robot vacuums.

## Literature Review
### Overview of YOLOv5
YOLO (You Only Look Once) is a state-of-the-art object detection algorithm. YOLOv5 offers significant improvements in lightweight architecture and detection accuracy.

### Robot Vacuum Case Studies
Existing products like iRobot's Roomba and LG CordZero primarily rely on IR sensors and cameras. Incorporating object detection can overcome their limitations.

## Data Acquisition
1. **Video Capture:** Use smartphones to record robot vacuum environments (e.g., living rooms, kitchens).
2. **DarkLabel Annotation:** Label objects such as chairs, tables, and obstacles in the collected data.

## Training on NVIDIA Jetson Nano
### Setup and Configuration
1. **Jetson Nano Configuration:** Install JetPack SDK, which includes CUDA, cuDNN, and TensorRT.
2. **Library Installation:**
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip
   pip3 install numpy torch torchvision
   pip3 install -r requirements.txt
   ```

3. **YOLOv5 Setup:** Clone the YOLOv5 repository and set up the environment.
   ```bash
   git clone https://github.com/ultralytics/yolov5
   ```

### Dataset Preparation
- Upload labeled datasets (annotated with DarkLabel) to Jetson Nano.
- Create a `data.yaml` file:
  ```yaml
  train: D:\yolov5-master\yolov5-master\data\train\images
  val: D:\yolov5-master\yolov5-master\data\valid\images
  test: D:\yolov5-master\yolov5-master\data\test\images

  nc: 1
  names: ['pixel']
  ```

### Model Training
1. Train YOLOv5 on Jetson Nano:
   ```bash
   python3 train.py --data data.yaml --cfg yolov5s.yaml --weights yolov5s.pt --batch-size 8 --epochs 20
   ```
   **Parameters:**
   - `--data`: Path to dataset configuration.
   - `--cfg`: Model configuration file (use `yolov5s.yaml` for lightweight models).
   - `--weights`: Pre-trained weights file.
   - `--batch-size`: Adjust for Jetson Nano's memory.
   - `--epochs`: Number of training iterations.

### Object Detection Execution
1. **Real-Time Detection with Webcam:**
   ```bash
   python3 detect.py --source 0 --weights runs/train/expX/weights/best.pt --img 640 --conf-thres 0.5
   ```
2. **Video File Detection:**
   ```bash
   python3 detect.py --source TEST1.mp4 --weights runs/train/expX/weights/best.pt --img 640 --conf-thres 0.5
   ```

## Evaluation Metrics
1. **Confusion Matrix**
2. **F1-Confidence Curve**
3. **Precision-Recall Curve**
4. **Precision-Confidence Curve**
5. **Labels Correlogram**

## Validation
1. **Testing on New Data:** Use `TEST1.mp4` to `TEST4.mp4` for validation.
2. **Real-Time Testing:** Verify detection accuracy using a webcam in real-world scenarios.
3. **Performance Summary:**
   - Objects detected with high confidence (e.g., chairs: 0.85+, bags: 0.9+).

## Limitations and Improvements
1. **Challenges:**
   - Dataset variability impacts detection accuracy.
   - FPS limitation (15-20) on Jetson Nano.

2. **Proposed Solutions:**
   - Data augmentation and additional training for diverse environments.
   - Optimize with TensorRT to improve FPS.
   - Expand object categories (e.g., cables, pets).

## Conclusion
The YOLOv5-based object detection system demonstrated high accuracy and real-time performance, making it suitable for applications like robot vacuums. With further data augmentation and optimization, the model can achieve even better performance in real-world scenarios.
