# binaries/list_cameras.py
import json
import platform
import cv2


def _open_probe(index: int) -> cv2.VideoCapture:
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)

def list_cameras(max_check: int = 5) -> list[dict]:
    cameras = []
    for i in range(max_check):
        cap = _open_probe(i)
        if cap.isOpened():
            # Get camera properties
            width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps    = cap.get(cv2.CAP_PROP_FPS)
            cameras.append({
                "index": i,
                "width": int(width),
                "height": int(height),
                "fps": int(fps),
                "label": f"Camera {i} ({int(width)}x{int(height)})"
            })
            cap.release()
    return cameras

if __name__ == "__main__":
    print(json.dumps(list_cameras()), flush=True)