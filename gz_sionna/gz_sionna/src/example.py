import cv2
import os

# Configuration
image_folder = 'scene/img3/'                # Folder containing images
output_video = 'cam3.mp4'   # Output video filename
frame_rate = 10                     # Frames per second

# Collect and sort image files
images = [img for img in os.listdir(image_folder) if img.startswith('scene_') and img.endswith('.png')]
images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort by scene number

# Read first image to get dimensions
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, _ = frame.shape

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# Write frames to video
for image_name in images:
    image_path = os.path.join(image_folder, image_name)
    frame = cv2.imread(image_path)
    if frame is not None:
        video.write(frame)

video.release()
print("✅ Video successfully created:", output_video)
