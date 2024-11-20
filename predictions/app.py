import streamlit as st
import cv2
import numpy as np
from datetime import timedelta
from helper import YOLO_Pred
import time

# Initialize YOLO_Pred with your model and YAML configuration
yolo = YOLO_Pred('hell/weights/best.onnx', 'data.yaml')

# Streamlit app configuration
st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.title("🔍 YOLO Object Detection")
st.write("Upload a video or image to detect objects.")

# Tabs for video and image detection
tab1, tab2 = st.tabs(["📹 Video Detection", "🖼️ Image Detection"])

# Video Detection
with tab1:
    st.subheader("Video Object Detection")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Save the uploaded video temporarily
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.success("Video uploaded! Processing...")

        # Load video
        video_file = cv2.VideoCapture(temp_video_path)

        # Create a placeholder for video display
        video_placeholder = st.empty()

        # Set display dimensions
        display_width = 640
        display_height = 360

        # Initialize frame counter and frame rate
        frame_count = 0
        fps = video_file.get(cv2.CAP_PROP_FPS)
        delay = 1 / fps  # Delay to make video play at correct speed

        # Flag to indicate if an emergency vehicle is detected
        emergency_detected = False

        # Create a placeholder for the detection message
        detection_message = st.empty()

        while True:
            # Read the next frame
            ret, frame = video_file.read()
            if not ret:
                st.write("End of video.")
                break

            # Get predictions
            img_pred, predicted_texts, boxes = yolo.predictions(frame)

            # Convert BGR to RGB
            img_pred_rgb = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)

            # Draw labels with color change based on condition (bounding box removed)
            for i, text in enumerate(predicted_texts):
                box = boxes[i]  # Assuming boxes are [x, y, w, h]
                x, y, w, h = box
                y += 35  # Shift the bounding box further below for more visibility

                # Set label color and font size based on object type
                color = (0, 0, 255) if "emergency" in text.lower() else (255, 0, 0)  # Red for emergency, blue otherwise
                font_size = 0.5  # Smaller font size for clarity
                font_thickness = 1  # Thinner font for better clarity
                font = cv2.FONT_HERSHEY_SIMPLEX  # Standard font

                # Draw filled background rectangle for text label
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_size, font_thickness)
                cv2.rectangle(img_pred_rgb, (x, y + h + 5), (x + text_width, y + h + text_height + 15), (0, 0, 0), -1)

                # Put text label below the bounding box
                cv2.putText(img_pred_rgb, text, (x, y + h + 20), font, font_size, (255, 255, 255), font_thickness)

                # Check if "emergency" is in the detected label
                if "emergency" in text.lower():
                    emergency_detected = True

            # Add timestamp with background for visibility
            timestamp = str(timedelta(seconds=int(frame_count / fps)))
            cv2.rectangle(img_pred_rgb, (5, 5), (170, 35), (0, 0, 0), -1)
            cv2.putText(img_pred_rgb, f'Time: {timestamp}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Resize the image for display
            img_resized = cv2.resize(img_pred_rgb, (display_width, display_height))

            # Show predicted image in placeholder
            video_placeholder.image(img_resized, caption='Predicted Video Frame', use_column_width=True)

            # Print "Emergency Vehicle Detected!" message if detected
            if emergency_detected:
                detection_message.text("🚨 Emergency Vehicle Detected!")
            else:
                detection_message.text("")  # Clear message if no emergency vehicle detected

            # Update frame counter
            frame_count += 1

            # Wait for the time to maintain frame rate
            time.sleep(delay)

            # Reset the flag for the next frame
            emergency_detected = False

        # Release video capture
        video_file.release()

# Image Detection
with tab2:
    st.subheader("Image Object Detection")
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Read and decode the image
        image = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Get predictions
        _, predicted_texts, boxes = yolo.predictions(img)

        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display the uploaded image without labels or boxes
        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

        # Display detected objects below the image
        if predicted_texts:
            st.write("### Detected Objects:")
            for i, (text, box) in enumerate(zip(predicted_texts, boxes)):
                x, y, w, h = box
                st.write(f"{i + 1}. **{text}** - Bounding Box: (x={x}, y={y}, w={w}, h={h})")
        else:
            st.warning("No objects detected. Try uploading a different image.")

# Sidebar Information
st.sidebar.write("### About")
st.sidebar.info(
    "This app uses the YOLO model for real-time object detection. Upload a video or image to see predictions in action."
)
