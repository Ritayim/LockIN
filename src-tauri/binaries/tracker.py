from __future__ import annotations

import base64
import platform
import cv2
import json
import sys
import time
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, asdict
from typing import Optional

# MediaPipe powers the iris-level eye-gaze branch. It's optional: if it
# can't be imported, the tracker still runs and the gaze classifier falls
# back to its head-pose pitch heuristic.
#
# We use the modern Tasks API (mediapipe.tasks.python.vision.FaceLandmarker)
# because the legacy `mediapipe.solutions` namespace is not shipped on
# Python 3.13+. The Tasks API still gives us the 478-point face mesh with
# refined iris landmarks (indices 468 / 473) when a face_landmarker.task
# model is provided.
try:
    import mediapipe as mp 
    from mediapipe.tasks.python import BaseOptions as _MpBaseOptions 
    _MEDIAPIPE_OK = True
except Exception as _mp_err: 
    mp = None 
    _MpBaseOptions = None 
    _mp_vision = None  
    _MEDIAPIPE_OK = False
    print(
        f"[tracker] mediapipe unavailable ({_mp_err}); "
        "falling back to head-pose pitch only.",
        file=sys.stderr,
        flush=True,
    )

# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Point2D:
    x: float
    y: float

@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

@dataclass
class EyeState:
    right: Point2D
    left: Point2D
    right_delta: Point2D
    left_delta: Point2D
    right_confidence: float
    left_confidence: float

@dataclass
class HeadState:
    position: Point2D
    delta: Point2D
    yaw: float    # negative = left, positive = right
    pitch: float  # negative = up, positive = down
    confidence: float

@dataclass
class NoseState:
    position: Point2D
    confidence: float

@dataclass
class IrisState:
    """
    Iris-based vertical eye gaze, derived from MediaPipe FaceMesh.

    pitch_left / pitch_right: normalized iris-Y inside the eyelid opening
        ~0.4–0.6 = looking at the monitor
        >  ~0.7  = looking down
        <  ~0.3  = looking up
    ear_left / ear_right: eye-aspect-ratio; low values mean the eye is
        squinting/closed and the pitch reading is meaningless.
    pitch:    average of the per-eye pitches that pass the EAR guard,
              or None when neither eye is reliable.
    """
    pitch: Optional[float]
    pitch_left: Optional[float]
    pitch_right: Optional[float]
    ear_left: Optional[float]
    ear_right: Optional[float]

@dataclass
class PersonState:
    present: bool
    count: int
    bbox: Optional[BoundingBox]

@dataclass
class TrackingFrame:
    person: PersonState
    head: Optional[HeadState]
    eyes: Optional[EyeState]
    nose: Optional[NoseState]
    iris: Optional[IrisState]
    timestamp_ms: int

# ─── Keypoint Indices (COCO 17-point) ─────────────────────────────────────────
KP_NOSE          = 0
KP_LEFT_EYE      = 1
KP_RIGHT_EYE     = 2
KP_LEFT_EAR      = 3
KP_RIGHT_EAR     = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER= 6

CONF_THRESHOLD   = 0.45
KP_THRESHOLD     = 0.50

# ─── MediaPipe FaceMesh landmark indices (refine_landmarks=True) ──────────────
# Iris centers (only present when refine_landmarks=True):
FM_IRIS_RIGHT       = 468
FM_IRIS_LEFT        = 473
# Eyelid landmarks used for iris-pitch + EAR. "Right" / "Left" refer to the
# subject's own right/left eye (which is how MediaPipe names them).
FM_RIGHT_EYE_UPPER  = 159
FM_RIGHT_EYE_LOWER  = 145
FM_RIGHT_EYE_INNER  = 133   # eye corner near the nose
FM_RIGHT_EYE_OUTER  = 33    # eye corner toward the temple
FM_LEFT_EYE_UPPER   = 386
FM_LEFT_EYE_LOWER   = 374
FM_LEFT_EYE_INNER   = 362
FM_LEFT_EYE_OUTER   = 263


# horizontal_ratio = |nose_x - left_eye_x| / |nose_x - right_eye_x|
LEFT_HORIZONTAL_THRESHOLD = 0.30
RIGHT_HORIZONTAL_THRESHOLD = 3.30

# pitch_ratio = (nose_y - eye_midpoint_y) / eye_span     (image-y grows down)
DOWN_PITCH_THRESHOLD = 0.5

# iris_pitch = (iris_y - upper_lid_y) / (lower_lid_y - upper_lid_y)
IRIS_DOWN_THRESHOLD = 0.4


def open_video_capture(index: int) -> cv2.VideoCapture:
    """Prefer DirectShow on Windows — MSMF often glitches with USB webcams."""
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return cap
        cap.release()
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    return cap


# ─── Smoother ─────────────────────────────────────────────────────────────────

class ExponentialSmoother:
    """Smooths noisy keypoint positions across frames."""
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha  # lower = smoother but more lag
        self.state: dict[str, float] = {}

    def smooth(self, key: str, value: float) -> float:
        if key not in self.state:
            self.state[key] = value
        else:
            self.state[key] = self.alpha * value + (1 - self.alpha) * self.state[key]
        return self.state[key]

    def smooth_point(self, key: str, x: float, y: float) -> Point2D:
        return Point2D(
            x=self.smooth(f"{key}_x", x),
            y=self.smooth(f"{key}_y", y),
        )

    def reset(self, key: str):
        self.state.pop(f"{key}_x", None)
        self.state.pop(f"{key}_y", None)

# ─── Tracker ──────────────────────────────────────────────────────────────────

class Tracker:
    def __init__(self, model_path: str, camera_index: int = 0,
                 face_landmarker_path: Optional[str] = None):
        self.smoother = ExponentialSmoother(alpha=0.4)
        self.start_time = time.time()
        self.prev_frame: Optional[TrackingFrame] = None
        self._face_landmarker_path = face_landmarker_path

        # Open camera first so failures are immediate; YOLO load can take tens of seconds.
        self.cap = open_video_capture(camera_index)
        if not self.cap.isOpened():
            print(
                f"Camera index {camera_index} not available; trying camera 0",
                file=sys.stderr,
                flush=True,
            )
            self.cap = open_video_capture(0)

        if not self.cap.isOpened():
            raise RuntimeError("No camera available")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.model = YOLO(model_path)

        # Optional iris tracker. Lives for the whole tracker lifetime
        # because the constructor allocates non-trivial graph resources.
        # We use the new Tasks API (FaceLandmarker) since it's the only
        # path available on Python 3.13+. It still returns the 478-point
        # mesh with refined iris landmarks when given the
        # face_landmarker.task model.
        self.face_landmarker = None
        self._iris_last_ts_ms = -1
        if _MEDIAPIPE_OK and self._face_landmarker_path:
            try:
                base = _MpBaseOptions(
                    model_asset_path=self._face_landmarker_path,
                )
                options = _mp_vision.FaceLandmarkerOptions(
                    base_options=base,
                    running_mode=_mp_vision.RunningMode.VIDEO,
                    num_faces=1,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.face_landmarker = (
                    _mp_vision.FaceLandmarker.create_from_options(options)
                )
            except Exception as e:  # pragma: no cover - env dependent
                self.face_landmarker = None
                print(
                    f"[tracker] FaceLandmarker init failed ({e}); "
                    "iris-based gaze disabled.",
                    file=sys.stderr,
                    flush=True,
                )
        elif _MEDIAPIPE_OK and not self._face_landmarker_path:
            print(
                "[tracker] FaceLandmarker model path not provided; "
                "iris-based gaze disabled.",
                file=sys.stderr,
                flush=True,
            )

    def timestamp(self) -> int:
        return int((time.time() - self.start_time) * 1000)

    def process_frame(self) -> tuple[Optional[TrackingFrame], Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None

        results = self.model(frame, verbose=False, conf=CONF_THRESHOLD)

        if not results or results[0].keypoints is None:
            self.prev_frame = TrackingFrame(
                person=PersonState(present=False, count=0, bbox=None),
                head=None,
                eyes=None,
                nose=None,
                iris=None,
                timestamp_ms=self.timestamp(),
            )
            return self.prev_frame, frame

        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints

        person_count = len(boxes)
        person_state = self._parse_person(boxes, person_count)
        head_state, eye_state, nose_state = self._parse_best_detection(keypoints, boxes)
        iris_state = self._extract_iris(frame)

        out = TrackingFrame(
            person=person_state,
            head=head_state,
            eyes=eye_state,
            nose=nose_state,
            iris=iris_state,
            timestamp_ms=self.timestamp(),
        )
        self.prev_frame = out
        return out, frame

    # ── Person ────────────────────────────────────────────────────────────────

    def _parse_person(self, boxes, count: int) -> PersonState:
        if count == 0:
            return PersonState(present=False, count=0, bbox=None)

        # Pick most confident detection
        confs = boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))
        box = boxes.xywh[best].cpu().numpy()

        bbox = BoundingBox(
            x=float(box[0] - box[2] / 2),
            y=float(box[1] - box[3] / 2),
            width=float(box[2]),
            height=float(box[3]),
        )

        return PersonState(present=True, count=count, bbox=bbox)

    # ── Head + Eyes ───────────────────────────────────────────────────────────

    def _parse_best_detection(self, keypoints, boxes):
        if keypoints is None or len(keypoints.xy) == 0:
            return None, None, None

        # Use most confident detection
        confs = boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))

        kp_xy  = keypoints.xy[best].cpu().numpy()   # shape: (17, 2)
        kp_conf = keypoints.conf[best].cpu().numpy() # shape: (17,)

        head_state = self._extract_head(kp_xy, kp_conf)
        eyes_state  = self._extract_eyes(kp_xy, kp_conf)
        nose_state = self._extract_nose(kp_xy, kp_conf)

        return head_state, eyes_state, nose_state

    def _extract_eyes(self, kp_xy, kp_conf) -> Optional[EyeState]:
        re_conf = kp_conf[KP_RIGHT_EYE]
        le_conf = kp_conf[KP_LEFT_EYE]

        if re_conf < KP_THRESHOLD or le_conf < KP_THRESHOLD:
            self.smoother.reset("right_eye")
            self.smoother.reset("left_eye")
            return None

        # Smooth raw keypoints
        right = self.smoother.smooth_point(
            "right_eye",
            kp_xy[KP_RIGHT_EYE][0],
            kp_xy[KP_RIGHT_EYE][1],
        )
        left = self.smoother.smooth_point(
            "left_eye",
            kp_xy[KP_LEFT_EYE][0],
            kp_xy[KP_LEFT_EYE][1],
        )

        # Compute deltas from previous frame
        prev_eyes = self.prev_frame.eyes if self.prev_frame else None
        if prev_eyes:
            right_delta = Point2D(
                x=right.x - prev_eyes.right.x,
                y=right.y - prev_eyes.right.y,
            )
            left_delta = Point2D(
                x=left.x - prev_eyes.left.x,
                y=left.y - prev_eyes.left.y,
            )
        else:
            right_delta = Point2D(0.0, 0.0)
            left_delta  = Point2D(0.0, 0.0)

        return EyeState(
            right=right,
            left=left,
            right_delta=right_delta,
            left_delta=left_delta,
            right_confidence=re_conf,
            left_confidence=le_conf,
        )

    def _extract_head(self, kp_xy, kp_conf) -> Optional[HeadState]:
        nose_conf = kp_conf[KP_NOSE]
        if nose_conf < KP_THRESHOLD:
            self.smoother.reset("head")
            return None

        position = self.smoother.smooth_point(
            "head",
            kp_xy[KP_NOSE][0],
            kp_xy[KP_NOSE][1],
        )

        prev_head = self.prev_frame.head if self.prev_frame else None
        delta = Point2D(
            x=position.x - prev_head.position.x if prev_head else 0.0,
            y=position.y - prev_head.position.y if prev_head else 0.0,
        )

        yaw   = self._estimate_yaw(kp_xy, kp_conf)
        pitch = self._estimate_pitch(kp_xy, kp_conf)

        return HeadState(position=position, delta=delta, yaw=yaw, pitch=pitch, confidence=nose_conf)

    def _extract_nose(self, kp_xy, kp_conf) -> Optional[NoseState]:
        nose_conf = kp_conf[KP_NOSE]
        if nose_conf < KP_THRESHOLD:
            return None

        position = self.smoother.smooth_point(
            "nose",
            kp_xy[KP_NOSE][0],
            kp_xy[KP_NOSE][1],
        )

        return NoseState(position=position, confidence=nose_conf)

    # ── Iris (MediaPipe FaceLandmarker / Tasks API) ───────────────────────────

    def _extract_iris(self, frame_bgr: np.ndarray) -> Optional[IrisState]:
        """
        Run MediaPipe FaceLandmarker on `frame_bgr` and compute a vertical
        iris position for each eye, normalized inside the eyelid opening.

        Returns None if the iris tracker is disabled or no face was found.
        Per-eye fields are None when the eye-aspect-ratio is too low for
        the iris reading to be meaningful (eye mostly closed).
        """
        if self.face_landmarker is None or frame_bgr is None:
            return None

        # FaceLandmarker.detect_for_video requires a strictly monotonically
        # increasing millisecond timestamp. Bump it ourselves if the wall-
        # clock timestamp would repeat (can happen at >1000 fps).
        ts_ms = self.timestamp()
        if ts_ms <= self._iris_last_ts_ms:
            ts_ms = self._iris_last_ts_ms + 1
        self._iris_last_ts_ms = ts_ms

        # MediaPipe expects RGB.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.face_landmarker.detect_for_video(mp_image, ts_ms)
        except Exception as e:  # pragma: no cover - defensive
            print(
                f"[tracker] FaceLandmarker.detect failed: {e}",
                file=sys.stderr, flush=True,
            )
            return None

        if not result or not result.face_landmarks:
            return IrisState(
                pitch=None, pitch_left=None, pitch_right=None,
                ear_left=None, ear_right=None,
            )

        # Tasks API: result.face_landmarks is a list-of-list-of NormalizedLandmark.
        landmarks = result.face_landmarks[0]
        h, w = frame_bgr.shape[:2]

        def xy(idx: int) -> tuple[float, float]:
            lm = landmarks[idx]
            return float(lm.x * w), float(lm.y * h)

        def per_eye(iris_idx: int, upper_idx: int, lower_idx: int,
                    inner_idx: int, outer_idx: int) -> tuple[Optional[float], float]:
            _, iris_y     = xy(iris_idx)
            _, upper_y    = xy(upper_idx)
            _, lower_y    = xy(lower_idx)
            inner_x, _    = xy(inner_idx)
            outer_x, _    = xy(outer_idx)

            lid_gap = lower_y - upper_y
            eye_width = abs(outer_x - inner_x)
            ear = float(lid_gap / max(eye_width, 1e-6))

            if lid_gap < 1.0 or ear < self.EAR_MIN:
                # Eye is squinted/closed: iris-Y inside such a thin opening
                # is essentially noise.
                return None, ear

            pitch = float((iris_y - upper_y) / lid_gap)
            return pitch, ear

        pitch_right, ear_right = per_eye(
            FM_IRIS_RIGHT, FM_RIGHT_EYE_UPPER, FM_RIGHT_EYE_LOWER,
            FM_RIGHT_EYE_INNER, FM_RIGHT_EYE_OUTER,
        )
        pitch_left, ear_left = per_eye(
            FM_IRIS_LEFT, FM_LEFT_EYE_UPPER, FM_LEFT_EYE_LOWER,
            FM_LEFT_EYE_INNER, FM_LEFT_EYE_OUTER,
        )

        valid = [p for p in (pitch_left, pitch_right) if p is not None]
        pitch = float(sum(valid) / len(valid)) if valid else None

        return IrisState(
            pitch=pitch,
            pitch_left=pitch_left,
            pitch_right=pitch_right,
            ear_left=ear_left,
            ear_right=ear_right,
        )

    def _estimate_yaw(self, kp_xy, kp_conf) -> float:
        """
        Estimate left/right head rotation from ear visibility.
        Negative = facing left, Positive = facing right.
        """
        nose_x   = kp_xy[KP_NOSE][0]
        lear_c   = kp_conf[KP_LEFT_EAR]
        rear_c   = kp_conf[KP_RIGHT_EAR]
        lear_x   = kp_xy[KP_LEFT_EAR][0]
        rear_x   = kp_xy[KP_RIGHT_EAR][0]

        if lear_c > KP_THRESHOLD and rear_c > KP_THRESHOLD:
            ear_mid_x = (lear_x + rear_x) / 2.0
            return float(np.clip((nose_x - ear_mid_x) * 2.0, -90, 90))
        elif rear_c > KP_THRESHOLD:
            return 45.0   # right ear only → facing right
        elif lear_c > KP_THRESHOLD:
            return -45.0  # left ear only → facing left
        return 0.0

    def _estimate_pitch(self, kp_xy, kp_conf) -> float:
        """
        Estimate up/down head tilt from nose-to-shoulder ratio.
        Negative = looking up, Positive = looking down.
        """
        lsh_c = kp_conf[KP_LEFT_SHOULDER]
        rsh_c = kp_conf[KP_RIGHT_SHOULDER]

        if lsh_c < KP_THRESHOLD or rsh_c < KP_THRESHOLD:
            return 0.0

        nose_y     = kp_xy[KP_NOSE][1]
        lsh_y      = kp_xy[KP_LEFT_SHOULDER][1]
        rsh_y      = kp_xy[KP_RIGHT_SHOULDER][1]
        shoulder_y = (lsh_y + rsh_y) / 2.0
        head_height = abs(shoulder_y - nose_y)

        if head_height < 1e-6:
            return 0.0

        return float(np.clip(((shoulder_y - nose_y) / head_height - 1.0) * 45.0, -90, 90))

 
    EAR_MIN = 0.15

    def gaze_info(self) -> dict:
        """
        Classify where the user is looking relative to the monitor and
        return the raw values used to make the decision. The raw values
        are exported so the UI can show them live for threshold tuning.

        Returned dict shape:
          {
            "state": "absent" | "down" | "away" | "straight",
            "pitch_ratio": float | None,        # head-pose pitch
            "horizontal_ratio": float | None,   # sideways
            "eyes_visible": "both" | "left" | "right" | "none",
            "iris_pitch": float | None,         # averaged iris-Y in eyelids
            "iris_left": float | None,
            "iris_right": float | None,
            "ear_left": float | None,
            "ear_right": float | None,
            "pitch_source": "eye" | "head" | None,  # what drove the decision
          }
        """
        info = {
            "state": "absent",
            "pitch_ratio": None,
            "horizontal_ratio": None,
            "eyes_visible": "none",
            "iris_pitch": None,
            "iris_left": None,
            "iris_right": None,
            "ear_left": None,
            "ear_right": None,
            "pitch_source": None,
        }

        pf = self.prev_frame
        if pf is None or not pf.person.present:
            return info
        if pf.head is None or pf.nose is None or pf.nose.confidence < KP_THRESHOLD:
            return info

        # Surface iris values up-front so the debug panel can show them
        # even in edge-case branches below.
        iris = pf.iris
        if iris is not None:
            info["iris_pitch"] = iris.pitch
            info["iris_left"] = iris.pitch_left
            info["iris_right"] = iris.pitch_right
            info["ear_left"] = iris.ear_left
            info["ear_right"] = iris.ear_right

        eyes = pf.eyes
        if eyes is None:
            info["state"] = "away"
            return info

        left_visible = eyes.left_confidence >= KP_THRESHOLD
        right_visible = eyes.right_confidence >= KP_THRESHOLD
        if left_visible and right_visible:
            info["eyes_visible"] = "both"
        elif left_visible:
            info["eyes_visible"] = "left"
        elif right_visible:
            info["eyes_visible"] = "right"

        if not left_visible or not right_visible:
            # Side-profile case: head + nose are detected but one eye is
            # occluded by the rest of the face. That's about as extreme
            # sideways as it gets.
            info["state"] = "away"
            return info

        nose = pf.nose
        nose_x, nose_y = nose.position.x, nose.position.y
        left_x, left_y = eyes.left.x, eyes.left.y
        right_x, right_y = eyes.right.x, eyes.right.y

        d_left = abs(nose_x - left_x)
        d_right = abs(nose_x - right_x)
        horizontal_ratio = float(d_left / (d_right + 1e-6))

        eye_midpoint_y = (left_y + right_y) / 2.0
        eye_span = abs(left_x - right_x) + 1e-6
        normalized_pitch = float((nose_y - eye_midpoint_y) / eye_span)

        info["pitch_ratio"] = normalized_pitch
        info["horizontal_ratio"] = horizontal_ratio

        # ── Down detection ─────────────────────────────────────────────
        # Primary: iris-based eye gaze when at least one eye gives us a
        # reliable iris reading (EAR guard already applied in _extract_iris).
        # Fallback: head-pose pitch_ratio.
        iris_pitch = iris.pitch if iris is not None else None
        if iris_pitch is not None:
            info["pitch_source"] = "eye"
            if iris_pitch < self.IRIS_DOWN_THRESHOLD:
                info["state"] = "down"
                return info
        else:
            info["pitch_source"] = "head"
            if normalized_pitch < self.DOWN_PITCH_THRESHOLD:
                info["state"] = "down"
                return info

        # ── Sideways detection (head-pose only) ────────────────────────
        # Only a hard sideways turn counts as "away". Everything else within
        # the wide horizontal band is treated as still facing the monitor.
        if (
            horizontal_ratio < self.LEFT_HORIZONTAL_THRESHOLD
            or horizontal_ratio > self.RIGHT_HORIZONTAL_THRESHOLD
        ):
            info["state"] = "away"
            return info

        info["state"] = "straight"
        return info

    def gaze_state(self) -> str:
        return self.gaze_info()["state"]

    def release(self):
        self.cap.release()
        if self.face_landmarker is not None:
            try:
                self.face_landmarker.close()
            except Exception:
                pass

# ─── Serialization ────────────────────────────────────────────────────────────

def to_dict(obj) -> dict:
    """Recursively convert dataclasses to JSON-serializable dicts."""
    if isinstance(obj, (Point2D, BoundingBox, EyeState, HeadState, NoseState,
                        IrisState, PersonState, TrackingFrame)):
        return {k: to_dict(v) for k, v in asdict(obj).items()}
    return obj


def json_sanitize(obj: object) -> object:
    """Turn NumPy scalars / arrays into plain Python so json.dumps never sees float32 etc."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return json_sanitize(obj.tolist())
    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, (int, float)):
        return obj
    return obj


def eye_movement_summary(eyes: dict) -> dict:
    """Average eye-keypoint motion → direction label for the UI."""
    rdx, rdy = eyes["right_delta"]["x"], eyes["right_delta"]["y"]
    ldx, ldy = eyes["left_delta"]["x"], eyes["left_delta"]["y"]
    dx = (rdx + ldx) / 2.0
    dy = (rdy + ldy) / 2.0
    mag = float(np.hypot(dx, dy))
    steady = mag < 1.2
    if steady:
        direction = "steady"
    else:
        adx, ady = abs(dx), abs(dy)
        if adx > ady * 1.15:
            direction = "right" if dx > 0 else "left"
        elif ady > adx * 1.15:
            direction = "down" if dy > 0 else "up"
        else:
            h = "right" if dx > 0 else "left"
            v = "down" if dy > 0 else "up"
            direction = f"{v}-{h}"
    return {
        "dx": float(dx),
        "dy": float(dy),
        "magnitude": float(mag),
        "direction": direction,
        "has_movement": not steady,
    }


def attach_eye_summary(payload: dict) -> dict:
    if payload.get("eyes"):
        payload["eye_movement"] = eye_movement_summary(payload["eyes"])
    else:
        payload["eye_movement"] = {
            "dx": 0.0,
            "dy": 0.0,
            "magnitude": 0.0,
            "direction": "no-eyes",
            "has_movement": False,
        }
    return payload


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    model_path           = sys.argv[1] if len(sys.argv) > 1 else "models/yolo26n-pose.pt"
    camera_index         = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    face_landmarker_path = sys.argv[3] if len(sys.argv) > 3 else None

    tracker = Tracker(
        model_path=model_path,
        camera_index=camera_index,
        face_landmarker_path=face_landmarker_path,
    )

    try:
        frame_idx = 0
        while True:
            frame, raw = tracker.process_frame()
            frame_idx += 1
            if frame is None:
                time.sleep(0.01)
                continue
            payload = attach_eye_summary(to_dict(frame))
            gaze = tracker.gaze_info()
            payload["gaze_state"] = gaze["state"]
            payload["gaze_debug"] = {
                "pitch_ratio": gaze["pitch_ratio"],
                "horizontal_ratio": gaze["horizontal_ratio"],
                "eyes_visible": gaze["eyes_visible"],
                "iris_pitch": gaze["iris_pitch"],
                "iris_left": gaze["iris_left"],
                "iris_right": gaze["iris_right"],
                "ear_left": gaze["ear_left"],
                "ear_right": gaze["ear_right"],
                "pitch_source": gaze["pitch_source"],
                "down_threshold": Tracker.DOWN_PITCH_THRESHOLD,
                "iris_threshold": Tracker.IRIS_DOWN_THRESHOLD,
                "ear_min": Tracker.EAR_MIN,
                "left_threshold": Tracker.LEFT_HORIZONTAL_THRESHOLD,
                "right_threshold": Tracker.RIGHT_HORIZONTAL_THRESHOLD,
            }
            
            if raw is not None and frame_idx % 4 == 0:
                h, w = raw.shape[:2]
                scale = min(1.0, 420.0 / max(w, h))
                small = cv2.resize(
                    raw,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
                ok, buf = cv2.imencode(
                    ".jpg",
                    small,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 52],
                )
                if ok:
                    payload["preview_jpeg_b64"] = base64.b64encode(buf).decode("ascii")
            print(json.dumps(json_sanitize(payload)), flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        tracker.release()

if __name__ == "__main__":
    main()