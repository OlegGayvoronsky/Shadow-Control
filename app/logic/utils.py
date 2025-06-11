import numpy as np

def extract_keypoints(results, type):
    non_arm_indices = [
        0,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32
    ]
    visibility_threshold = 0.45

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        def process_landmark(lm):
            if lm.visibility < visibility_threshold:
                return [0.0, 0.0, 0.0, 0.0]
            return [(-lm.x + 1) / 2, (-lm.y + 1) / 2, (-lm.z + 1) / 2, lm.visibility]

        if type == 1:
            # Используем все 33 точки
            pose = np.array([process_landmark(lm) for lm in landmarks]).flatten()

        elif type == 2:
            pose = np.array([process_landmark(landmarks[i]) for i in non_arm_indices]).flatten()

    else:
        # В зависимости от типа подставляем нули
        if type == 1:
            pose = np.zeros(33 * 4)
        else:
            pose = np.zeros(len(non_arm_indices) * 4)

    return pose
