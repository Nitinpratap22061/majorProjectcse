import streamlit as st
import cv2
import numpy as np
from datetime import timedelta
from helper import YOLO_Pred
from twilio.rest import Client  # Twilio for SMS
import time

# Initialize YOLO_Pred with your model and YAML configuration
yolo = YOLO_Pred('predictions/hell/weights/best.onnx', 'predictions/data.yaml')

# Twilio Configuration
TWILIO_ACCOUNT_SID = 'AC25c5fc49466cde96a344070b25cb7b2d'  
TWILIO_AUTH_TOKEN = 'e5eeabdc261c9502425c7eb14ef3f3ed'   
TWILIO_PHONE_NUMBER = '+16814343845'  
RECIPIENT_PHONE_NUMBER = '+916205815679'  

# Function to send SMS
def send_sms(to_phone, message_body):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    try:
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=to_phone
        )
        st.write(f"📱 SMS sent successfully to {to_phone}: {message.sid}")
    except Exception as e:
        st.error(f"❌ Failed to send SMS: {e}")

# Camera calibration values (adjust these based on your camera setup)
focal_length = 800  # Example value; it depends on your camera setup
real_world_width = 2.5  # Real-world width of the object (e.g., an average ambulance)

# Function to estimate distance based on bounding box width
def calculate_distance(bbox_width, focal_length, real_width):
    if bbox_width > 0:
        distance = (focal_length * real_width) / bbox_width
        return distance
    return float('inf')

# Streamlit app configuration
st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.title("🔍 Emergency Vehicle Detection")
st.write("Upload a video or image to detect objects.")

# Tabs for video and image detection
tab1, tab2 = st.tabs(["📹 Video Detection", "🖼️ Image Detection"])

# Store detected vehicles and their distances
detected_vehicles = set()

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
        fps = video_file.get(cv2.CAP_PROP_FPS) # how many frames are displayed per second:
        delay = 1 / fps  # Delay to make video play at correct speed

        message_sent = False  # Reset message flag for each new video

        while True:
            # Read the next frame
            ret, frame = video_file.read() #ret bool val frame read sucessfully or not ''frame:-numpy array frame video
            if not ret:
                st.write("End of video.")
                break

            # Get predictions
            img_pred, predicted_texts, boxes = yolo.predictions(frame) #img_pred: NumPy array of the frame with bounding boxes drawn predicted_texts: ["emergency : 98%", "car : 95%"] boxes: [[120, 150, 200, 300], [50, 60, 100, 150]]

            # Convert BGR to RGB
            img_pred_rgb = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)

            for i, text in enumerate(predicted_texts):
                box = boxes[i]  # Assuming boxes are [x, y, w, h]
                x, y, w, h = box

                # Use box coordinates as a unique identifier for the detected object
                object_id = f"{x}-{y}-{w}-{h}"

                label, confidence_str = text.split(' : ')
                confidence = int(confidence_str.replace('%', '').strip())

                # Calculate distance
                distance = calculate_distance(w, focal_length, real_world_width)

                # Draw the label directly on the object with smaller font size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0  # Slightly smaller font size
                color = (0, 0, 0)  # Black color for better contrast
                thickness = 2

                # Position the text directly inside the bounding box (within the object)
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                text_x = x + (w - text_size[0]) // 2
                text_y = y + h // 2  # Place text vertically in the middle of the object

                # Draw the text directly on the object
                cv2.putText(img_pred_rgb, label, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

                # If it's an emergency vehicle, send the SMS and display the message
                if label.lower() == 'emergency' and object_id not in detected_vehicles:
                    detected_vehicles.add(object_id)

                    # Notify in Streamlit
                    st.write(f"🚨 **Emergency Vehicle Detected!** Distance: {distance:.2f} meters")

                    # Send SMS only for emergency vehicles
                    if not message_sent:
                        send_sms(RECIPIENT_PHONE_NUMBER, f"Emergency Vehicle Detected! Distance: {distance:.2f} meters")
                        message_sent = True

                # For non-emergency vehicles, just mark them
                elif label.lower() != 'emergency' and object_id not in detected_vehicles:
                    detected_vehicles.add(object_id)
                    st.write(f"🚗 **Non-Emergency Vehicle Detected!** Distance: {distance:.2f} meters")

            # Resize the image for display
            img_resized = cv2.resize(img_pred_rgb, (display_width, display_height))

            # Show predicted image in placeholder
            video_placeholder.image(img_resized, caption='Predicted Video Frame', use_column_width=True)

            # Update frame counter
            frame_count += 1

            # Wait for the time to maintain frame rate
            time.sleep(delay)

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

        # Display the uploaded image
        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

        # Display detected objects
        if predicted_texts:
            st.write("### Detected Objects:")
            message_sent = False  # Reset message flag for each image

            for i, (text, box) in enumerate(zip(predicted_texts, boxes)):
                x, y, w, h = box
                label, confidence_str = text.split(' : ')
                confidence = int(confidence_str.replace('%', '').strip())

                # Calculate distance
                distance = calculate_distance(w, focal_length, real_world_width)

                # Draw the label directly on the object with smaller font size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0  # Slightly smaller font size
                color = (0, 0, 0)  # Black color for better contrast
                thickness = 2

                # Position the text directly inside the bounding box (within the object)
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                text_x = x + (w - text_size[0]) // 2
                text_y = y + h // 2  # Place text vertically in the middle of the object

                # Draw the text directly on the object
                cv2.putText(img_rgb, label, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

                # Display messages based on the vehicle type
                if label.lower() == 'emergency':
                    st.write(f"{i + 1}. **{label}** - Distance: {distance:.2f} meters")

                    # Send SMS only for emergency vehicles
                    if not message_sent:
                        send_sms(RECIPIENT_PHONE_NUMBER, f"Emergency Vehicle Detected! Distance: {distance:.2f} meters")
                        message_sent = True
                else:
                    st.write(f"{i + 1}. **{label}** - Non-Emergency Vehicle Detected (SMS Not Sent)")

        # Display the image with text
        st.image(img_rgb, caption="Detected Image with Text", use_column_width=True)

# Sidebar Information
st.sidebar.write("### About")
st.sidebar.info(
    "A smart system designed to identify emergency vehicles like ambulances, fire trucks, and police cars in real-time using machine learning."
)
