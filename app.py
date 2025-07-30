import streamlit as st
from pathlib import Path
from ultralytics import YOLO
import tempfile
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


st.set_page_config(page_title="AI Soccer Analyzer", layout="wide")

st.title("AI Soccer Analyzer")

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
        if st.button("Analyze Video"):
            # --- Video Processing ---
            with st.spinner("Analyzing video... This may take a few minutes."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                video_path_out = tfile.name

                cap = cv2.VideoCapture(str(selected_video_path))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(video_path_out, fourcc, fps, (width, height))
                
                player_positions = []
                ball_positions = []
                player_counts_over_time = []
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.track(frame, persist=True, tracker="botsort.yaml", classes=[0, 32]) # Track person and sports ball
                    
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
                            # Assume one ball, take the first one
                            x_center, y_center, w, h = boxes[ball_indices[0]]
                            ball_positions.append((x_center, y_center))

                    
                    player_counts_over_time.append(num_players)
                    out.write(annotated_frame)
                    frame_count += 1
                
                cap.release()
                out.release()
                
                st.video(video_path_out)
                tfile.close()

                st.success("Analysis complete!")
            
            # --- Report Generation ---
            st.header("Post-Game Analysis Report")
            
            # Heatmap
            if player_positions:
                with st.spinner("Generating heatmap..."):
                    heatmap_fig = generate_heatmap(player_positions, width, height)
                    if heatmap_fig:
                        st.subheader("Player Position Heatmap")
                        st.pyplot(heatmap_fig, use_container_width=True)
            
            # Ball Trajectory
            if ball_positions:
                with st.spinner("Generating ball trajectory..."):
                    ball_fig = generate_ball_trajectory(ball_positions, width, height)
                    if ball_fig:
                        st.subheader("Ball Trajectory Map")
                        st.pyplot(ball_fig, use_container_width=True)

            # Player count chart
            if player_counts_over_time:
                with st.spinner("Generating player count chart..."):
                    st.subheader("Player Count Over Time")
                    chart_data = pd.DataFrame({
                        "Frame": range(len(player_counts_over_time)),
                        "Number of Players": player_counts_over_time
                    }).set_index("Frame")
                    st.line_chart(chart_data) 