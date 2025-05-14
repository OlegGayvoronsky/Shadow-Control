import numpy as np

def extract_keypoints(results):
    pose = np.array([[(-res.x + 1) / 2, (-res.y + 1) / 2, (-res.z + 1) / 2, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return pose
