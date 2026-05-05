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

@dataclass
class HeadState:
    position: Point2D
    delta: Point2D
    yaw: float    # negative = left, positive = right
    pitch: float  # negative = up, positive = down

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
    def __init__(self, model_path: str, camera_index: int = 0):
        self.smoother = ExponentialSmoother(alpha=0.4)
        self.start_time = time.time()
        self.prev_eyes: Optional[EyeState] = None
        self.prev_head: Optional[HeadState] = None

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

    def timestamp(self) -> int:
        return int((time.time() - self.start_time) * 1000)

    def process_frame(self) -> tuple[Optional[TrackingFrame], Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None

        results = self.model(frame, verbose=False, conf=CONF_THRESHOLD)

        if not results or results[0].keypoints is None:
            self.prev_eyes = None
            self.prev_head = None
            return TrackingFrame(
                person=PersonState(present=False, count=0, bbox=None),
                head=None,
                eyes=None,
                timestamp_ms=self.timestamp(),
            ), frame

        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints

        person_count = len(boxes)
        person_state = self._parse_person(boxes, person_count)
        head_state, eye_state = self._parse_best_detection(keypoints, boxes)

        return TrackingFrame(
            person=person_state,
            head=head_state,
            eyes=eye_state,
            timestamp_ms=self.timestamp(),
        ), frame

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
            return None, None

        # Use most confident detection
        confs = boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))

        kp_xy  = keypoints.xy[best].cpu().numpy()   # shape: (17, 2)
        kp_conf = keypoints.conf[best].cpu().numpy() # shape: (17,)

        head_state = self._extract_head(kp_xy, kp_conf)
        eye_state  = self._extract_eyes(kp_xy, kp_conf)

        self.prev_head = head_state
        self.prev_eyes = eye_state

        return head_state, eye_state

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
        if self.prev_eyes:
            right_delta = Point2D(
                x=right.x - self.prev_eyes.right.x,
                y=right.y - self.prev_eyes.right.y,
            )
            left_delta = Point2D(
                x=left.x - self.prev_eyes.left.x,
                y=left.y - self.prev_eyes.left.y,
            )
        else:
            right_delta = Point2D(0.0, 0.0)
            left_delta  = Point2D(0.0, 0.0)

        return EyeState(
            right=right,
            left=left,
            right_delta=right_delta,
            left_delta=left_delta,
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

        delta = Point2D(
            x=position.x - self.prev_head.position.x if self.prev_head else 0.0,
            y=position.y - self.prev_head.position.y if self.prev_head else 0.0,
        )

        yaw   = self._estimate_yaw(kp_xy, kp_conf)
        pitch = self._estimate_pitch(kp_xy, kp_conf)

        return HeadState(position=position, delta=delta, yaw=yaw, pitch=pitch)

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

    def release(self):
        self.cap.release()

# ─── Serialization ────────────────────────────────────────────────────────────

def to_dict(obj) -> dict:
    """Recursively convert dataclasses to JSON-serializable dicts."""
    if isinstance(obj, (Point2D, BoundingBox, EyeState, HeadState, PersonState, TrackingFrame)):
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
    model_path   = sys.argv[1] if len(sys.argv) > 1 else "models/yolo26n-pose.pt"
    camera_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    tracker = Tracker(model_path=model_path, camera_index=camera_index)

    try:
        frame_idx = 0
        while True:
            frame, raw = tracker.process_frame()
            frame_idx += 1
            if frame is None:
                time.sleep(0.01)
                continue
            payload = attach_eye_summary(to_dict(frame))
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