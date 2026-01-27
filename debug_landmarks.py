import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os

# Create a simple test face image with known proportions
# Or use an existing image in the system
img_path = "/workspace/facial-uniqueness/test.jpg"

# Check if we have any uploaded images in gradio cache
import glob
gradio_imgs = glob.glob("/workspace/facial-uniqueness/.gradio/**/*.jpg", recursive=True) + \
              glob.glob("/workspace/facial-uniqueness/.gradio/**/*.png", recursive=True)

if gradio_imgs:
    img_path = gradio_imgs[-1]
    print(f"Using uploaded image: {img_path}")
else:
    # Download using curl which handles redirects better
    import subprocess
    result = subprocess.run([
        "curl", "-L", "-o", "/tmp/test_face.jpg",
        "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&w=400"
    ], capture_output=True)
    img_path = "/tmp/test_face.jpg"
    print(f"Downloaded test image")

# Initialize InsightFace
app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load and analyze
img = cv2.imread(img_path)
if img is None:
    print(f"Failed to load image: {img_path}")
    exit(1)
    
faces = app.get(img)

if faces:
    face = faces[0]
    lmk = face.landmark_2d_106
    
    print(f"Total landmarks: {len(lmk)}")
    print(f"Image size: {img.shape}")
    
    # Find extremes
    x_coords = lmk[:, 0]
    y_coords = lmk[:, 1]
    
    print(f"\nX range: {x_coords.min():.1f} - {x_coords.max():.1f}")
    print(f"Y range: {y_coords.min():.1f} - {y_coords.max():.1f}")
    
    # Compute face dimensions
    face_width = x_coords.max() - x_coords.min()
    face_height = y_coords.max() - y_coords.min()
    print(f"\nFace width: {face_width:.1f}")
    print(f"Face height: {face_height:.1f}")
    print(f"Height/Width ratio: {face_height/face_width:.3f}")
    
    # Find key points
    left_idx = np.argmin(x_coords)
    right_idx = np.argmax(x_coords)
    top_idx = np.argmin(y_coords)
    bottom_idx = np.argmax(y_coords)
    
    print(f"\nExtreme points:")
    print(f"  Leftmost: index {left_idx}")
    print(f"  Rightmost: index {right_idx}")  
    print(f"  Topmost: index {top_idx}")
    print(f"  Bottommost (chin): index {bottom_idx}")
    
    # Identify face contour - should be indices 0-32 or thereabouts
    # The contour typically goes around the face edge
    center_x = (x_coords.min() + x_coords.max()) / 2
    center_y = (y_coords.min() + y_coords.max()) / 2
    
    # Cluster analysis - find groups of points
    print("\n=== LANDMARK GROUPS (by region) ===")
    
    # Group by vertical position
    third_h = face_height / 3
    top_y = y_coords.min()
    
    upper = [i for i in range(len(lmk)) if y_coords[i] < top_y + third_h]
    middle = [i for i in range(len(lmk)) if top_y + third_h <= y_coords[i] < top_y + 2*third_h]
    lower = [i for i in range(len(lmk)) if y_coords[i] >= top_y + 2*third_h]
    
    print(f"Upper third (eyes/brows): {sorted(upper)}")
    print(f"Middle third (nose): {sorted(middle)}")
    print(f"Lower third (mouth/chin): {sorted(lower)}")
    
    # Within upper region, split left/right for eyes
    upper_left = [i for i in upper if x_coords[i] < center_x]
    upper_right = [i for i in upper if x_coords[i] >= center_x]
    
    print(f"\nUpper-left (left eye/brow): {sorted(upper_left)}")
    print(f"Upper-right (right eye/brow): {sorted(upper_right)}")
    
    # Find eyes by looking for oval clusters
    # Eyes are typically indices 60-75 in 106 model
    print("\n=== CHECKING DOCUMENTED EYE INDICES 60-75 ===")
    for i in range(60, 76):
        print(f"  [{i}] x={lmk[i][0]:.1f} y={lmk[i][1]:.1f}")
    
    left_eye_60_67 = lmk[60:68]
    right_eye_68_75 = lmk[68:76]
    
    left_eye_center = left_eye_60_67.mean(axis=0)
    right_eye_center = right_eye_68_75.mean(axis=0)
    
    eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
    eye_spacing_ratio = eye_distance / face_width
    
    print(f"\nLeft eye center (60-67): {left_eye_center}")
    print(f"Right eye center (68-75): {right_eye_center}")
    print(f"Eye distance: {eye_distance:.1f}")
    print(f"Eye spacing ratio: {eye_spacing_ratio:.3f}")
    
else:
    print("No face detected")
