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
    
    # Draw all landmarks with index numbers
    vis = img.copy()
    for i, (x, y) in enumerate(lmk):
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
        cv2.putText(vis, str(i), (int(x)+2, int(y)-2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)
    
    cv2.imwrite("/workspace/facial-uniqueness/landmarks_labeled.jpg", vis)
    print("Saved landmarks_labeled.jpg")
    
    # Also identify by examining clusters
    # The 106 points follow JD-landmark format:
    # 0-32: face contour (33 points)
    # 33-37: left eyebrow (5 points) 
    # 38-42: right eyebrow (5 points)
    # 43-46: nose bridge (4 points)
    # 47-51: nose bottom (5 points)
    # 52-57: left eye (6 points)
    # 58-63: right eye (6 points)
    # 64-67: upper outer lip (4 points)
    # 68-71: lower outer lip (4 points)  
    # 72-82: upper/lower inner lip (11 points)
    # 83-95: mouth interior (13 points)
    # 96-105: additional points
    
    # Let me verify by checking the positions
    print("\n=== Testing JD-landmark format hypothesis ===")
    
    # Face contour 0-32
    contour = lmk[0:33]
    print(f"Contour (0-32): x range {contour[:,0].min():.0f}-{contour[:,0].max():.0f}")
    
    # Left eyebrow 33-37
    left_brow = lmk[33:38]
    print(f"Left brow (33-37): center at {left_brow.mean(axis=0)}")
    
    # Right eyebrow 38-42
    right_brow = lmk[38:43]
    print(f"Right brow (38-42): center at {right_brow.mean(axis=0)}")
    
    # Check indices 52-57 for left eye
    test_left_eye = lmk[52:58]
    print(f"Testing 52-57 as left eye: center at {test_left_eye.mean(axis=0)}")
    
    # Check indices 58-63 for right eye
    test_right_eye = lmk[58:64]
    print(f"Testing 58-63 as right eye: center at {test_right_eye.mean(axis=0)}")
    
    # Check if these make sense relative to face width
    face_width = lmk[:,0].max() - lmk[:,0].min()
    left_center = test_left_eye.mean(axis=0)
    right_center = test_right_eye.mean(axis=0)
    eye_dist = np.linalg.norm(right_center - left_center)
    
    print(f"\nEye distance: {eye_dist:.1f}")
    print(f"Face width: {face_width:.1f}")
    print(f"Ratio: {eye_dist/face_width:.3f}")
