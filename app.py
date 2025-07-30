import streamlit as st
from pathlib import Path
from ultralytics import YOLO
import tempfile
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


st.set_page_config(page_title="AI Soccer Analyzer", layout="wide")

st.title("AI Soccer Analyzer")

# --- Constants ---
# Lower resolution to 480p for cloud compatibility
TARGET_WIDTH = 854

# --- Model and Functions ---
@st.cache_resource
def load_model():
    """Loads the YOLOv8 model."""
    return YOLO("yolov8n.pt")

def generate_heatmap(positions, width, height, cmap="hot"):
    """Generates a heatmap from position data."""
    pos_array = np.array(positions)
    if pos_array.size == 0:
        return None
        
    x = pos_array[:, 0]
    y = pos_array[:, 1]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.kdeplot(x=x, y=y, cmap=cmap, shade=True, thresh=0.05, n_levels=30, ax=ax)
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis() # Match video coordinates (0,0 is top-left)
    ax.set_facecolor('#2E5528') # Soccer field green
    ax.set_title("Player Position Heatmap", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    return fig

def generate_ball_trajectory(positions, width, height):
    """Generates a trajectory map for the ball."""
    pos_array = np.array(positions)
    if pos_array.size < 2:
        return None
        
    x = pos_array[:, 0]
    y = pos_array[:, 1]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x, y, color='white', marker='o', linestyle='-', markersize=4, markerfacecolor='yellow')
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()
    ax.set_facecolor('#2E5528')
    ax.set_title("Ball Trajectory Map", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    return fig

model = load_model()

# --- UI Setup ---
video_dir = Path("data/raw")
video_files = list(video_dir.glob("*.mp4"))
video_file_names = [file.name for file in video_files]

selected_video_name = st.selectbox("Choose a video to analyze:", video_file_names)

if selected_video_name:
    selected_video_path = video_dir / selected_video_name
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Video")
        st.video(str(selected_video_path))

    with col2:
        st.subheader("Analyzed Video")
        
        # Placeholder for the video
        video_display = st.empty()

        if st.button("Analyze Video"):
            video_path_out = None
            try:
                # --- Video Processing ---
                with st.spinner("Analyzing video... This will be faster now!"):
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    video_path_out = tfile.name
                    tfile.close()

                    cap = cv2.VideoCapture(str(selected_video_path))
                    
                    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    aspect_ratio = original_height / original_width
                    target_height = int(TARGET_WIDTH * aspect_ratio)

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    out = cv2.VideoWriter(video_path_out, fourcc, fps, (TARGET_WIDTH, target_height))
                    
                    player_positions = []
                    ball_positions = []
                    player_counts_over_time = []

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        resized_frame = cv2.resize(frame, (TARGET_WIDTH, target_height))
                        results = model.track(resized_frame, persist=True, tracker="botsort.yaml", classes=[0, 32])
                        annotated_frame = results[0].plot()

                        num_players = 0
                        if results[0].boxes.id is not None:
                            boxes = results[0].boxes.xywh.cpu().numpy()
                            classes = results[0].boxes.cls.cpu().numpy()
                            
                            player_indices = np.where(classes == 0)[0]
                            num_players = len(player_indices)
                            for i in player_indices:
                                x_center, y_center, w, h = boxes[i]
                                player_positions.append((x_center, y_center + h / 2))
                                
                            ball_indices = np.where(classes == 32)[0]
                            if len(ball_indices) > 0:
                                x_center, y_center, w, h = boxes[ball_indices[0]]
                                ball_positions.append((x_center, y_center))
                        
                        player_counts_over_time.append(num_players)
                        out.write(annotated_frame)
                    
                    cap.release()
                    out.release()
                    st.success("Analysis complete!")

                # --- Display Video ---
                video_display.video(video_path_out)
                
                # --- Report Generation ---
                st.header("Post-Game Analysis Report")
                
                if player_positions:
                    with st.spinner("Generating heatmap..."):
                        heatmap_fig = generate_heatmap(player_positions, TARGET_WIDTH, target_height)
                        if heatmap_fig:
                            st.subheader("Player Position Heatmap")
                            st.pyplot(heatmap_fig, use_container_width=True)
                
                if ball_positions:
                    with st.spinner("Generating ball trajectory..."):
                        ball_fig = generate_ball_trajectory(ball_positions, TARGET_WIDTH, target_height)
                        if ball_fig:
                            st.subheader("Ball Trajectory Map")
                            st.pyplot(ball_fig, use_container_width=True)

                if player_counts_over_time:
                    with st.spinner("Generating player count chart..."):
                        st.subheader("Player Count Over Time")
                        chart_data = pd.DataFrame({
                            "Frame": range(len(player_counts_over_time)),
                            "Number of Players": player_counts_over_time
                        }).set_index("Frame")
                        st.line_chart(chart_data)

            finally:
                # Clean up the temp file if it exists
                if video_path_out and os.path.exists(video_path_out):
                    os.unlink(video_path_out) 