import cv2
import numpy as np
from insightface.app import FaceAnalysis
import subprocess

subprocess.run(["curl", "-L", "-o", "/tmp/test_face.jpg", "-s",
    "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&w=400"])

app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

img = cv2.imread("/tmp/test_face.jpg")
faces = app.get(img)

if faces:
    face = faces[0]
    lmk = face.landmark_2d_106
    
    x_coords = lmk[:, 0]
    y_coords = lmk[:, 1]
    
    face_width = x_coords.max() - x_coords.min()
    face_height = y_coords.max() - y_coords.min()
    
    # Categorize all 106 points
    regions = {
        "LEFT_BROW": [], "RIGHT_BROW": [], 
        "LEFT_EYE": [], "RIGHT_EYE": [],
        "NOSE": [], "MOUTH": [], "CHIN": [], "CONTOUR": []
    }
    
    for i in range(106):
        nx = (lmk[i][0] - x_coords.min()) / face_width
        ny = (lmk[i][1] - y_coords.min()) / face_height
        
        # Check if on face edge (contour)
        if nx < 0.1 or nx > 0.9:
            regions["CONTOUR"].append(i)
        elif ny < 0.2:  # Top - brows
            if nx < 0.5:
                regions["LEFT_BROW"].append(i)
            else:
                regions["RIGHT_BROW"].append(i)
        elif ny < 0.4:  # Upper-mid - eyes
            if nx < 0.45:
                regions["LEFT_EYE"].append(i)
            elif nx > 0.55:
                regions["RIGHT_EYE"].append(i)
            else:
                regions["NOSE"].append(i)  # nose bridge between eyes
        elif ny < 0.7:  # Mid - nose
            regions["NOSE"].append(i)
        elif ny < 0.9:  # Lower - mouth
            regions["MOUTH"].append(i)
        else:  # Bottom - chin
            regions["CHIN"].append(i)
    
    print("=== LANDMARK REGIONS ===\n")
    for region, indices in regions.items():
        print(f"{region}: {indices}")
    
    # Calculate proper eye spacing
    if regions["LEFT_EYE"] and regions["RIGHT_EYE"]:
        left_eye_pts = lmk[regions["LEFT_EYE"]]
        right_eye_pts = lmk[regions["RIGHT_EYE"]]
        
        left_center = left_eye_pts.mean(axis=0)
        right_center = right_eye_pts.mean(axis=0)
        
        eye_dist = np.linalg.norm(right_center - left_center)
        
        # Use actual face width from contour
        print(f"\n=== EYE SPACING ===\n")
        print(f"Left eye center: {left_center}")
        print(f"Right eye center: {right_center}")
        print(f"Inter-pupillary distance: {eye_dist:.1f}")
        print(f"Face width (bizygomatic): {face_width:.1f}")
        print(f"IPD / Face Width ratio: {eye_dist/face_width:.3f}")
