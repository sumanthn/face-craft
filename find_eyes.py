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
    
    left_eye_candidates = []
    right_eye_candidates = []
    
    for i in range(106):
        nx = (lmk[i][0] - x_coords.min()) / face_width
        ny = (lmk[i][1] - y_coords.min()) / face_height
        
        if 0.25 < ny < 0.45:
            if nx < 0.4:
                left_eye_candidates.append(i)
            elif nx > 0.6:
                right_eye_candidates.append(i)
    
    print(f"Left eye indices: {left_eye_candidates}")
    print(f"Right eye indices: {right_eye_candidates}")
    
    left_eye_pts = lmk[left_eye_candidates]
    right_eye_pts = lmk[right_eye_candidates]
    
    left_center = left_eye_pts.mean(axis=0)
    right_center = right_eye_pts.mean(axis=0)
    
    eye_dist = np.linalg.norm(right_center - left_center)
    ratio = eye_dist / face_width
    
    print(f"Eye distance: {eye_dist:.1f}, Face width: {face_width:.1f}")
    print(f"Eye spacing ratio: {ratio:.3f}")
