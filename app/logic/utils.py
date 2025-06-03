import numpy as np

def extract_keypoints(results, type):
    non_arm_indices = [
        0,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32
    ]
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        if type == 1:
            # Используем все 33 точки
            pose = np.array([
                [(-res.x + 1) / 2, (-res.y + 1) / 2, (-res.z + 1) / 2, res.visibility]
                for res in landmarks
            ]).flatten()

        elif type == 2:
            pose = np.array([
                [(-landmarks[i].x + 1) / 2, (-landmarks[i].y + 1) / 2, (-landmarks[i].z + 1) / 2, landmarks[i].visibility]
                for i in non_arm_indices
            ]).flatten()
    else:
        # В зависимости от типа подставляем размер нуля
        if type == 1:
            pose = np.zeros(33 * 4)
        else:
            pose = np.zeros(len(non_arm_indices) * 4)

    return pose
