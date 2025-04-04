import cv2
import numpy as np
import os
import time
import mediapipe as mp
from train_model_utils import extract_keypoints


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=False,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils


# Создаем объект для записи видео
frame_size = (640, 480)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")


def setup_folders(DATA_PATH, actions, no_sequences):
    for action in actions:
        try:
            os.makedirs(os.path.join(DATA_PATH, action))
        except:
            pass

        dirmax = len(os.listdir(os.path.join(DATA_PATH, action)))
        for sequence in range(1, no_sequences + 1):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(dirmax + sequence)))
            except:
                pass

def collect_keypoints(actions, start_folder, no_sequences):
    cap = cv2.VideoCapture(0)
    for action in actions:
        for sequence in range(start_folder, start_folder + no_sequences):
            vid_path = os.path.join(DATA_PATH, action, str(sequence), "video.mp4")
            video_writer = cv2.VideoWriter(vid_path, fourcc, 30, frame_size)
            for frame_num in range(sequence_length):
                ret, frame = cap.read()

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                video_writer.write(frame)

                if frame_num == 0:
                    cv2.putText(frame, '{}'.format(action), (0, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (150, 224, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, '{}/{}'.format(sequence - 1, no_sequences), (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 224, 0), 4, cv2.LINE_AA)

                    cv2.imshow("Pose Detection", frame)
                    cv2.waitKey(1000)
                else:
                    cv2.putText(frame, '{}/{}'.format(sequence - 1, no_sequences), (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 224, 0), 4, cv2.LINE_AA)
                    cv2.imshow("Pose Detection", frame)

                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    DATA_PATH = os.path.join('VidData')
    # try:
    #     os.makedirs(DATA_PATH)
    # except:
    #     pass

    actions = np.array(["walking forward", "walking backward", "walking left", "walking right", "running forward", "running back", "sit down", "jump", "one-handed weapon attack", "two-handed weapon attack", "shield block", "weapon block", "attacking magic", "bowstring pull", "nothing"])
    no_sequences = 50
    sequence_length = 30
    start_folder = 1

    # setup_folders(DATA_PATH, actions, no_sequences)
    #
    # collect_keypoints(actions, start_folder, no_sequences)
