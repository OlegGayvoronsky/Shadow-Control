import numpy as np
import cv2

def create_segment(segment):
    target_fps = 30
    fps = len(segment)
    segment = np.array(segment)
    indices = []


    if fps > target_fps:
        indices = np.linspace(0, fps - 1, target_fps, dtype=int)
        resampled_segment = segment[indices]
    elif target_fps > fps > 1:
        resampled_segment = []
        add_count = target_fps - fps
        for i, joints in enumerate(segment):
            if i > 0 and add_count > 0:
                _joints = np.ndarray([17, 2], dtype=int)
                add_count -= 1
                for j in range(17):
                    _x, _y = joints[j - 1]
                    x, y = joints[j]
                    _joints[j][0] = (max(_x, x) + min(_x, x)) / 2
                    _joints[j][1] = (max(_y, y) + min(_y, y)) / 2
                indices.append(i)
                resampled_segment.append(joints)
            indices.append(i)
            resampled_segment.append(joints)
        resampled_segment = np.array(resampled_segment)
    else:
        resampled_segment = segment

    return resampled_segment, indices


def show(frames, orig_imgs,  ids, video_writer):

    images = np.array(orig_imgs)[ids, :, :]

    for f, img in zip(frames, images):
        h, w, _ = img.shape
        for joint in f:
            x, y = joint
            cv2.circle(img, (int(x * w), int(y * h)), 5, (0, 0, 255), -1)
        img = np.array(img, dtype=np.uint8)  # Преобразуем в формат uint8 (если нужно)
        video_writer.write(img)

    cv2.destroyAllWindows()
