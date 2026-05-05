# LockIN

Desktop app that shows a live webcam preview and estimates **eye movement**, **head pose** (yaw / pitch), and **person presence** using a YOLO pose model. The UI is built with **React** and **TypeScript**; tracking runs in a **Python** subprocess orchestrated by **Tauri 2** (Rust).

## Requirements

| Component | Notes |
|-----------|--------|
| **Node.js** | For the Vite frontend and `npm run tauri` |
| **Rust** | Stable toolchain with `cargo` (for Tauri) |
| **Python 3** | |
| **Webcam** | USB or built-in camera |

Install Python dependencies used by the tracker and camera listing:

```bash
pip install -r src-tauri/binaries/requirements.txt
```

That installs `ultralytics`, `opencv-python`, `numpy`, and `pyinstaller` (optional for packaging scripts).

## Model weights

Tracking expects a YOLO pose checkpoint at:

`src-tauri/models/yolo26n-pose.pt`

The repository is set up with this path; if you replace the file, keep the same name or update `main.rs` to match.

## Development

From the repository root:

```bash
npm install
npm run tauri dev
```

This starts the Vite dev server and opens the Tauri window. The first run may take a while while Ultralytics loads the model.

## Production build

```bash
npm run tauri build
```

Artifacts follow the usual Tauri output layout for your platform.

## How it works

1. **Camera list** — Rust runs `src-tauri/binaries/list_cameras.py`, which probes indices with OpenCV and prints JSON.
2. **Tracking** — `start_tracking` spawns `src-tauri/binaries/eye_tracker.py` with the model path and camera index. The script writes one JSON object per line on stdout (preview as base64 JPEG inside the payload).
3. **Events** — The shell reads stdout/stderr and emits:
   - `tracking-frame` — parsed frame JSON
   - `tracker-error` — stderr lines from Python
   - `tracker-stopped` — when the reader finishes or the process ends

Frontend code listens in `src/tracker.ts` and renders in `src/App.tsx`.

## Troubleshooting

- **“Camera list failed” or no cameras** — Grant camera permission, try another index, or on Windows ensure no other app exclusively locks the device.
- **Stuck on “Starting Python tracker”** — Confirm `python` / `py` works in a terminal and that `pip install -r src-tauri/binaries/requirements.txt` succeeded. First model load can take 30–90 seconds.
- **Tracker errors in the status line** — Read the message from `tracker-error`; missing model file or wrong Python environment are common causes.

## Project structure

Omitted: `node_modules/`, `dist/`, `src-tauri/target/`, and other generated or local-only paths.

```
LockIN/
├── index.html
├── package.json
├── package-lock.json
├── tsconfig.json
├── tsconfig.node.json
├── vite.config.ts
├── LICENSE
├── public/
│   └── tauri.svg
├── src/
│   ├── App.tsx
│   ├── App.css
│   ├── main.tsx
│   ├── tracker.ts
│   ├── vite-env.d.ts
│   ├── components/
│   │   └── WebcamCapture.tsx
│   └── constants/
│       └── RustHandler.tsx
└── src-tauri/
    ├── build.rs
    ├── Cargo.toml
    ├── Cargo.lock
    ├── tauri.conf.json
    ├── capabilities/
    │   └── default.json
    ├── binaries/
    │   ├── eye_tracker.py
    │   ├── list_cameras.py
    │   └── requirements.txt
    ├── models/
    │   └── yolo26n-pose.pt
    └── src/
        ├── lib.rs
        ├── main.rs
        └── webcam.rs
```

### Files

| Path | Role |
|------|------|
| `src/App.tsx` | Main screen: camera picker, start/stop, live preview, eye/head/person metrics. |
| `src/tracker.ts` | Types for tracking payloads; `invoke` calls (`list_cameras`, `start_tracking`, `stop_tracking`); listeners for `tracking-frame`, `tracker-error`, `tracker-stopped`. |
| `src/constants/RustHandler.tsx` | Shared string constant for IPC naming (minimal). |
| `src-tauri/src/main.rs` | **Desktop binary:** manages the Python tracker process, exposes Tauri commands, forwards stdout/stderr to frontend events. |
| `src-tauri/src/lib.rs` | Tauri **library** entry (minimal builder); used for targets that link the app as a library (e.g. mobile); day-to-day desktop flow is `main.rs`. |
| `src-tauri/binaries/eye_tracker.py` | Opens the webcam, runs YOLO pose, computes eye movement and head pose; prints **one JSON object per line** on stdout (including base64 JPEG preview). |
| `src-tauri/binaries/list_cameras.py` | Probes camera indices with OpenCV; prints a **JSON array** of devices (used on startup to fill the dropdown). |

