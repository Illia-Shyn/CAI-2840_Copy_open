# Computer Vision System Overview

This document details the computer vision (CV) components of the AI Soccer Analyzer project. The system is designed to process soccer match videos to detect, track, and analyze the movement of players and the ball.

---

## 1. Core Technologies

- **Object Detection Model:** **YOLOv8n**
  - We use the "nano" version of the YOLOv8 model, pre-trained on the COCO dataset. This model was chosen for its excellent balance of speed and accuracy, making it suitable for real-time analysis on standard hardware and for deployment on cloud platforms like Streamlit Cloud.
  - The model is loaded directly from the `ultralytics` library using `YOLO("yolov8n.pt")`. The library automatically handles the download and caching of the model weights.

- **Object Tracking Algorithm:** **BoT-SORT (ByteTrack with SORT)**
  - To track objects across frames, we utilize the BoT-SORT algorithm, which is integrated directly into the `ultralytics` library.
  - It is invoked via the `model.track(..., tracker="botsort.yaml")` method. BoT-SORT is effective at maintaining unique IDs for each detected object, even through brief occlusions, which is crucial for tracking individual players.

## 2. System Configuration & Implementation

The entire CV pipeline is managed within the `app.py` script.

- **Object Class Filtering:**
  - The system is configured to specifically track two object classes from the COCO dataset:
    - **Class `0`:** `person` (Represents the players)
    - **Class `32`:** `sports ball` (Represents the soccer ball)
  - This is configured in the tracking call: `model.track(..., classes=[0, 32])`. Filtering ensures that the model only spends resources tracking relevant objects and that our analysis is not cluttered with irrelevant detections.

- **Data Extraction:**
  - For each frame of the video, the system processes the results from the `model.track()` method.
  - It extracts the following information:
    1.  **Bounding Boxes (`boxes.xywh`):** The coordinates and dimensions of each detected player and the ball.
    2.  **Class IDs (`boxes.cls`):** The class of each detected object (either 0 or 32).
    3.  **Tracker IDs (`boxes.id`):** The unique ID assigned to each object by the BoT-SORT tracker.

## 3. CV-Powered Analytics

The extracted coordinate data is used to generate the following analytical reports:

- **Player Position Heatmap:**
  - For each detected player, the **bottom-center** coordinate of their bounding box (`x_center, y_center + h/2`) is stored. This point is a better proxy for a player's position on the field than the center of the bounding box.
  - After processing the entire video, a 2D kernel density estimate (KDE) plot is generated using `seaborn.kdeplot`. This visualizes the areas of the field where players spent the most time.

- **Ball Trajectory Map:**
  - The center coordinate of the ball's bounding box is recorded for each frame.
  - These coordinates are then plotted as a connected line chart, creating a clear visual map of the ball's path throughout the video clip.

- **Player Count Over Time:**
  - In each frame, the system counts the number of detected `person` objects.
  - This data is used to generate a line chart that shows how the number of visible players changes over the course of the video. 