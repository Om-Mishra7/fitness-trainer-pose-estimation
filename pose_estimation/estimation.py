import cv2
import os
import urllib.request
import mediapipe as mp

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'pose_landmarker_full.task'
)
_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task'
)


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print(f'Downloading pose landmarker model to {_MODEL_PATH} ...')
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print('Model downloaded.')


class _LandmarkContainer:
    """Wraps a landmark list so callers can use .landmark[idx] (old API style)."""
    __slots__ = ('landmark',)

    def __init__(self, landmark_list):
        self.landmark = landmark_list


class _PoseResults:
    """Makes the new Tasks API result look like the old mp.solutions result."""
    __slots__ = ('pose_landmarks',)

    def __init__(self, result):
        if result.pose_landmarks:
            self.pose_landmarks = _LandmarkContainer(result.pose_landmarks[0])
        else:
            self.pose_landmarks = None


class PoseEstimator:
    def __init__(self, static_mode=False, model_complexity=1):
        _ensure_model()
        self._ts_ms = 0
        running_mode = VisionTaskRunningMode.IMAGE if static_mode else VisionTaskRunningMode.VIDEO
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=running_mode,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.pose = PoseLandmarker.create_from_options(options)
        self._video_mode = not static_mode

    def close(self):
        if self.pose:
            self.pose.close()
            self.pose = None

    def estimate_pose(self, frame, exercise_type):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        if self._video_mode:
            self._ts_ms += 33  # ~30 fps
            raw = self.pose.detect_for_video(mp_image, self._ts_ms)
        else:
            raw = self.pose.detect(mp_image)

        results = _PoseResults(raw)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            if exercise_type == 'squat':
                self.draw_squat_lines(frame, lm)
            elif exercise_type == 'push_up':
                self.draw_push_up_lines(frame, lm)
            elif exercise_type == 'hammer_curl':
                self.draw_hammerl_curl_lines(frame, lm)

        return results

    def draw_hammerl_curl_lines(self, frame, landmarks):
        shoulder_right = [int(landmarks[11].x * frame.shape[1]), int(landmarks[11].y * frame.shape[0])]
        elbow_right    = [int(landmarks[13].x * frame.shape[1]), int(landmarks[13].y * frame.shape[0])]
        wrist_right    = [int(landmarks[15].x * frame.shape[1]), int(landmarks[15].y * frame.shape[0])]
        shoulder_left  = [int(landmarks[12].x * frame.shape[1]), int(landmarks[12].y * frame.shape[0])]
        elbow_left     = [int(landmarks[14].x * frame.shape[1]), int(landmarks[14].y * frame.shape[0])]
        wrist_left     = [int(landmarks[16].x * frame.shape[1]), int(landmarks[16].y * frame.shape[0])]
        cv2.line(frame, shoulder_left,  elbow_left,  (0, 0, 255), 4, 2)
        cv2.line(frame, elbow_left,     wrist_left,  (0, 0, 255), 4, 2)
        cv2.line(frame, shoulder_right, elbow_right, (0, 0, 255), 4, 2)
        cv2.line(frame, elbow_right,    wrist_right, (0, 0, 255), 4, 2)

    def draw_squat_lines(self, frame, landmarks):
        hip          = [int(landmarks[23].x * frame.shape[1]), int(landmarks[23].y * frame.shape[0])]
        knee         = [int(landmarks[25].x * frame.shape[1]), int(landmarks[25].y * frame.shape[0])]
        shoulder     = [int(landmarks[11].x * frame.shape[1]), int(landmarks[11].y * frame.shape[0])]
        hip_right    = [int(landmarks[24].x * frame.shape[1]), int(landmarks[24].y * frame.shape[0])]
        knee_right   = [int(landmarks[26].x * frame.shape[1]), int(landmarks[26].y * frame.shape[0])]
        shoulder_right = [int(landmarks[12].x * frame.shape[1]), int(landmarks[12].y * frame.shape[0])]
        cv2.line(frame, shoulder,       hip,        (178, 102, 255), 2)
        cv2.line(frame, hip,            knee,       (178, 102, 255), 2)
        cv2.line(frame, shoulder_right, hip_right,  (51, 153, 255), 2)
        cv2.line(frame, hip_right,      knee_right, (51, 153, 255), 2)

    def draw_push_up_lines(self, frame, landmarks):
        shoulder_left  = [int(landmarks[11].x * frame.shape[1]), int(landmarks[11].y * frame.shape[0])]
        elbow_left     = [int(landmarks[13].x * frame.shape[1]), int(landmarks[13].y * frame.shape[0])]
        wrist_left     = [int(landmarks[15].x * frame.shape[1]), int(landmarks[15].y * frame.shape[0])]
        shoulder_right = [int(landmarks[12].x * frame.shape[1]), int(landmarks[12].y * frame.shape[0])]
        elbow_right    = [int(landmarks[14].x * frame.shape[1]), int(landmarks[14].y * frame.shape[0])]
        wrist_right    = [int(landmarks[16].x * frame.shape[1]), int(landmarks[16].y * frame.shape[0])]
        cv2.line(frame, shoulder_left,  elbow_left,  (0, 0, 255), 2)
        cv2.line(frame, elbow_left,     wrist_left,  (0, 0, 255), 2)
        cv2.line(frame, shoulder_right, elbow_right, (102, 0, 0), 2)
        cv2.line(frame, elbow_right,    wrist_right, (102, 0, 0), 2)
