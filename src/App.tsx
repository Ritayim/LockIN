import { useCallback, useEffect, useRef, useState } from "react";
import {
  getCameras,
  startTrackingSession,
  stopTracking,
  type Camera,
  type TrackingFrame,
} from "./tracker";
import "./App.css";

export default function App() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selected, setSelected] = useState(0);
  const [tracking, setTracking] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [frame, setFrame] = useState<TrackingFrame | null>(null);

  const cleanupRef = useRef<(() => void) | null>(null);
  const slowHintRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearSlowHint = useCallback(() => {
    if (slowHintRef.current !== null) {
      window.clearTimeout(slowHintRef.current);
      slowHintRef.current = null;
    }
  }, []);

  const safeCleanup = useCallback(() => {
    clearSlowHint();
    cleanupRef.current?.();
    cleanupRef.current = null;
  }, [clearSlowHint]);

  useEffect(() => {
    getCameras()
      .then((cams) => {
        setCameras(cams);
        if (cams.length > 0) setSelected(cams[0].index);
      })
      .catch((e) => setStatus(`Camera list failed: ${e}`));
  }, []);

  useEffect(() => {
    return () => {
      void stopTracking();
      safeCleanup();
    };
  }, [safeCleanup]);

  const handleStart = async () => {
    if (tracking) return;
    safeCleanup();
    setTracking(true);
    setStatus(
      "Starting Python tracker… If this sits here, check Python is on PATH (py, python) and ultralytics/OpenCV install.",
    );
    setFrame(null);

    slowHintRef.current = window.setTimeout(() => {
      setStatus(
        "Still working — loading YOLO pose model on first run can take 30–90s. Preview appears after the first frame.",
      );
    }, 8000);

    try {
      const { cleanup } = await startTrackingSession(
        selected,
        (f) => {
          clearSlowHint();
          setFrame((prev) => ({
            ...f,
            preview_jpeg_b64: f.preview_jpeg_b64 ?? prev?.preview_jpeg_b64,
          }));
          setStatus("Live — eye movement updates automatically");
        },
        (err) => {
          clearSlowHint();
          setStatus(`Tracker: ${err}`);
        },
        () => {
          setTracking(false);
          setStatus("Stopped");
          setFrame(null);
          safeCleanup();
        },
      );
      cleanupRef.current = cleanup;
    } catch (e) {
      setTracking(false);
      setStatus(`Error: ${e}`);
      safeCleanup();
    }
  };

  const handleStop = async () => {
    if (!tracking && !cleanupRef.current) return;
    try {
      await stopTracking();
    } catch (e) {
      setStatus(`Stop error: ${e}`);
    }
    setTracking(false);
    setStatus("Stopped");
    setFrame(null);
    safeCleanup();
  };

  const em = frame?.eye_movement;
  const head = frame?.head;

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>LockIN</h1>
          <p className="subtitle">Camera view with automatic eye-movement detection</p>
        </div>
        <div className="toolbar">
          <label className="field">
            <span>Camera</span>
            <select
              value={selected}
              onChange={(e) => setSelected(Number(e.target.value))}
              disabled={tracking}
            >
              {cameras.length === 0 ? (
                <option value={0}>No cameras found</option>
              ) : (
                cameras.map((cam) => (
                  <option key={cam.index} value={cam.index}>
                    {cam.label}
                  </option>
                ))
              )}
            </select>
          </label>
          <button
            type="button"
            className="btn primary"
            onClick={handleStart}
            disabled={tracking || cameras.length === 0}
          >
            Start
          </button>
          <button
            type="button"
            className="btn"
            onClick={handleStop}
            disabled={!tracking}
          >
            Stop
          </button>
        </div>
      </header>

      <p className="status">{status}</p>

      <div className="layout">
        <section className="preview-card">
          {frame?.preview_jpeg_b64 ? (
            <img
              className="preview"
              alt="Live camera"
              src={`data:image/jpeg;base64,${frame.preview_jpeg_b64}`}
            />
          ) : (
            <div className="preview placeholder">
              {tracking
                ? "Waiting for video…"
                : "Press Start to open the camera and run the tracker."}
            </div>
          )}
        </section>

        <aside className="metrics">
          <div className="metric">
            <h2>Eye movement</h2>
            <div
              className={`direction-pill dir-${(em?.direction ?? "steady").replace(/[^a-z0-9-]/gi, "")}`}
            >
              {em?.direction ?? "—"}
            </div>
            <dl className="dl">
              <div>
                <dt>Movement</dt>
                <dd>{em?.has_movement ? "Yes" : "No"}</dd>
              </div>
              <div>
                <dt>Speed</dt>
                <dd>{em ? em.magnitude.toFixed(2) : "—"} px/frame</dd>
              </div>
              <div>
                <dt>Δx / Δy</dt>
                <dd>
                  {em
                    ? `${em.dx.toFixed(1)} / ${em.dy.toFixed(1)}`
                    : "—"}
                </dd>
              </div>
            </dl>
          </div>

          <div className="metric">
            <h2>Head pose</h2>
            <dl className="dl">
              <div>
                <dt>Yaw</dt>
                <dd>{head ? `${head.yaw.toFixed(1)}°` : "—"}</dd>
              </div>
              <div>
                <dt>Pitch</dt>
                <dd>{head ? `${head.pitch.toFixed(1)}°` : "—"}</dd>
              </div>
            </dl>
          </div>

          <div className="metric">
            <h2>Scene</h2>
            <dl className="dl">
              <div>
                <dt>People</dt>
                <dd>{frame?.person.count ?? "—"}</dd>
              </div>
              <div>
                <dt>Person visible</dt>
                <dd>{frame?.person.present ? "Yes" : "No"}</dd>
              </div>
            </dl>
          </div>
        </aside>
      </div>
    </div>
  );
}
