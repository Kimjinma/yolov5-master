# YOLOv5-Based Object Detection for Real-Time Object Recognition

## Project Overview
This project utilizes YOLOv5, a deep learning-based object detection model, to analyze video data and implement real-time object detection. The technology identifies and tracks specific objects (e.g., blankets, furniture) efficiently and accurately within video frames.

### Key Steps:
1. **Video Data Collection:** Gather videos such as `TEST1.mp4`, `TEST2.mp4`, etc.
2. **YOLOv5 Model Training:** Improve detection accuracy using a pre-trained YOLOv5 model.
   ![image](https://github.com/user-attachments/assets/44fa0732-61aa-4162-b30d-6e39d9d270b6)

4. **Object Detection:** Analyze videos to store and visualize detected object information.
   ![image](https://github.com/user-attachments/assets/01cb4224-8e08-43bb-bf93-ce6ab153db28)

5. **Webcam Detection:** Perform real-time object detection from live video feeds.
   ![image](https://github.com/user-attachments/assets/d83a74a1-da9d-41fd-91eb-683a13ae973a)
   ![image](https://github.com/user-attachments/assets/c88832a7-fb14-4009-b7d2-7ed3754b82c4)
![image](https://github.com/user-attachments/assets/ad5a0244-6b48-4689-bc0b-74c8b4b22a9f)


## Background Information
### Necessity of Robot Vacuums
Robot vacuums have become essential in reducing domestic chores. Efficient obstacle avoidance and focused cleaning on specific areas are critical features.

### Role of Object Detection
Traditional robot vacuums rely on basic sensors, limiting their ability to recognize obstacles or navigate complex environments effectively. YOLOv5’s object detection capabilities enable adaptive and intelligent behaviors in robot vacuums.

### YOLOv5 Features
YOLOv5 is a lightweight model suitable for real-time object detection. It operates efficiently on both GPUs and CPUs, making it adaptable for hardware-limited devices like robot vacuums.

## Project Objectives
1. **Obstacle and Object Detection:** Design a system enabling a robot vacuum to detect obstacles and specific objects (e.g., chairs, tables, toys).
2. **Dynamic Navigation:** Facilitate adaptive cleaning paths and focused cleaning around detected objects.

## Project Phases
1. **Data Collection:** Record images and videos from diverse environments where robot vacuums operate.
2. **Data Labeling:** Use tools like DarkLabel to annotate objects in the dataset.
3. **YOLOv5 Model Training:** Train YOLOv5 using the labeled dataset.
4. **Object Detection Implementation:** Deploy the trained model to detect objects in videos and live streams.

## Project Strengths
1. **Real-Time Obstacle Avoidance:** Efficient real-time detection and path optimization through YOLOv5.
2. **Targeted Object Detection:** Enable focused cleaning or avoidance around specific objects (e.g., toys, cables).
3. **Broad Applicability:** Enhances robot vacuum performance in households, offices, factories, and more.

## Project Significance
1. **Advancing Smart Homes:** Aligns with IoT and AI trends in smart home environments.
2. **Improving Efficiency:** Outperforms traditional sensor-based robot vacuums in navigation and cleaning efficiency.
3. **Industry Potential:** Applicable to warehouse and logistics robots beyond vacuum cleaners.

## Challenges
1. **Data Scarcity:** Limited datasets with diverse environments and objects may restrict model performance.
2. **Model Optimization:** YOLOv5 requires additional optimization for hardware-constrained platforms like robot vacuums.

## Literature Review
### Overview of YOLOv5
YOLO (You Only Look Once) is a leading object detection algorithm. YOLOv5 features significant improvements in efficiency and accuracy.

### Robot Vacuum Case Studies
Existing products like iRobot’s Roomba and LG CordZero rely on IR sensors and cameras. Incorporating object detection addresses their limitations.

## Data Acquisition
1. **Video Capture:** Use smartphones to record environments (e.g., living rooms, kitchens) for robot vacuums.
2. **DarkLabel Annotation:** Annotate objects such as chairs, tables, and obstacles in the collected videos.
   ![image](https://github.com/user-attachments/assets/af7ea716-5cc2-41a2-beea-a0f0bbdc425d)


## Training on NVIDIA Jetson Nano
 Setup and Configuration
1. **Jetson Nano Configuration:** Install JetPack SDK, including CUDA, cuDNN, and TensorRT.
2. **Library Installation:**
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip
   pip3 install numpy torch torchvision
   pip3 install -r requirements.txt
   ```

3. **YOLOv5 Setup:** Clone the YOLOv5 repository and configure the environment.
   ```bash
   git clone https://github.com/ultralytics/yolov5
   ```

 Dataset Preparation
- Upload labeled datasets (annotated with DarkLabel) to Jetson Nano.
- Create a `data.yaml` file:
  ```yaml
  train: D:\yolov5-master\yolov5-master\data\train\images
  val: D:\yolov5-master\yolov5-master\data\valid\images
  test: D:\yolov5-master\yolov5-master\data\test\images

  nc: 1
  names: ['pixel']
  ```

 Model Training
Train YOLOv5 on Jetson Nano:
   ```bash
   python3 train.py --data data.yaml --cfg yolov5s.yaml --weights yolov5s.pt --batch-size 8 --epochs 20
   ```
   **Parameters:**
   - `--data`: Path to dataset configuration.
   - `--cfg`: Model configuration file (e.g., `yolov5s.yaml`).
   - `--weights`: Pre-trained weights file.
   - `--batch-size`: Adjust for Jetson Nano’s memory capacity.
   - `--epochs`: Number of training iterations.

### Object Detection Execution
1. **Real-Time Detection with Webcam:
   ```bash
   python3 detect.py --source 0 --weights runs/train/expX/weights/best.pt --img 640 --conf-thres 0.5
   ```
2. **Video File Detection:
   ```bash
   python3 detect.py --source TEST1.mp4 --weights runs/train/expX/weights/best.pt --img 640 --conf-thres 0.5
   ```

## Evaluation Metrics
1. **Confusion Matrix
   ![image](https://github.com/user-attachments/assets/51fb1060-c686-4033-ba0e-d28d763f3016)

2. **F1-Confidence Curve
   ![image](https://github.com/user-attachments/assets/aa6a47a4-1057-43e4-b48a-3c58cceec625)

3. **Precision-Recall Curve
 ![image](https://github.com/user-attachments/assets/9c93c11e-e3ae-4f12-941e-b0db41c1b7cf)

4. **Precision-Confidence Curve**
 ![image](https://github.com/user-attachments/assets/108b809f-60b8-4db8-803e-c3aa83f3aa7d)

5. **Labels Correlogram**
 ![image](https://github.com/user-attachments/assets/2082ef72-fe84-4655-8363-300f79c7429c)

6. **results**
![image](https://github.com/user-attachments/assets/b95dcf10-d220-4695-9f2a-5442a3e829a8)

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
