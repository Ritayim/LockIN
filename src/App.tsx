import { useCallback, useEffect, useRef, useState } from "react";
import {
  isPermissionGranted,
  requestPermission,
  sendNotification,
} from "@tauri-apps/plugin-notification";
import {
  getCameras,
  startTrackingSession,
  stopTracking,
  type Camera,
  type GazeState,
  type TrackingFrame,
} from "./tracker";
import "./App.css";

const GAZE_LABELS: Record<GazeState, string> = {
  straight: "Looking at monitor",
  down: "Looking down",
  away: "Looking away",
  absent: "No face detected",
};

// How long a non-"straight" gaze must be sustained before we react. The
// frame stream is noisy (a stray detection blink shouldn't reset the timer),
// so we require this many consecutive ms of "not straight" before it counts.
const AWAY_DEBOUNCE_MS = 400;

// Tick interval for the live "away for Xs" counter. Independent of the
// frame rate so the UI stays smooth even if frames are sparse.
const COUNTER_TICK_MS = 250;

async function ensureNotificationPermission(): Promise<boolean> {
  try {
    if (await isPermissionGranted()) return true;
    return (await requestPermission()) === "granted";
  } catch (e) {
    console.error("Notification permission check failed:", e);
    return false;
  }
}

async function fireAwayNotification(state: GazeState, awaySec: number) {
  const body =
    state === "down"
      ? `You've been looking down for ${awaySec}s. Eyes up — get back on track.`
      : state === "away"
      ? `You've been looking away for ${awaySec}s. Refocus on the monitor.`
      : `Camera hasn't seen you for ${awaySec}s. Are you still here?`;

  try {
    if (await ensureNotificationPermission()) {
      sendNotification({ title: "LockIN — refocus", body });
    }
  } catch (e) {
    console.error("Failed to send system notification:", e);
  }
}

export default function App() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selected, setSelected] = useState(0);
  const [tracking, setTracking] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [frame, setFrame] = useState<TrackingFrame | null>(null);

  const [thresholdSec, setThresholdSec] = useState(60);
  const [awayMs, setAwayMs] = useState(0);
  const [alertActive, setAlertActive] = useState(false);

  // Refs that drive the away state machine; refs (not state) so the frame
  // handler doesn't need to be re-created on every tick.
  const awaySinceRef = useRef<number | null>(null);
  const notifiedRef = useRef(false);
  const lastGazeRef = useRef<GazeState>("absent");
  const thresholdRef = useRef(thresholdSec);
  useEffect(() => {
    thresholdRef.current = thresholdSec;
  }, [thresholdSec]);

  const cleanupRef = useRef<(() => void) | null>(null);
  const slowHintRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tickerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearSlowHint = useCallback(() => {
    if (slowHintRef.current !== null) {
      window.clearTimeout(slowHintRef.current);
      slowHintRef.current = null;
    }
  }, []);

  const clearTicker = useCallback(() => {
    if (tickerRef.current !== null) {
      window.clearInterval(tickerRef.current);
      tickerRef.current = null;
    }
  }, []);

  const resetAwayState = useCallback(() => {
    awaySinceRef.current = null;
    notifiedRef.current = false;
    setAwayMs(0);
    setAlertActive(false);
  }, []);

  const safeCleanup = useCallback(() => {
    clearSlowHint();
    clearTicker();
    cleanupRef.current?.();
    cleanupRef.current = null;
    resetAwayState();
  }, [clearSlowHint, clearTicker, resetAwayState]);

  useEffect(() => {
    getCameras()
      .then((cams) => {
        setCameras(cams);
        if (cams.length > 0) setSelected(cams[0].index);
      })
      .catch((e) => setStatus(`Camera list failed: ${e}`));
    void ensureNotificationPermission();
  }, []);

  useEffect(() => {
    return () => {
      void stopTracking();
      safeCleanup();
    };
  }, [safeCleanup]);

  const handleFrame = useCallback(
    (f: TrackingFrame) => {
      lastGazeRef.current = f.gaze_state;

      if (f.gaze_state === "straight") {
        // Eyes back on the monitor: clear the away streak and arm the
        // notifier again for the next time they drift.
        awaySinceRef.current = null;
        notifiedRef.current = false;
      } else if (awaySinceRef.current === null) {
        awaySinceRef.current = Date.now();
      }
    },
    [],
  );

  // Live ticker — drives the displayed "away for Xs", and is what actually
  // fires the notification when the threshold is crossed. Decoupling this
  // from the frame stream means we still alert correctly if frames stall.
  useEffect(() => {
    if (!tracking) return;
    tickerRef.current = window.setInterval(() => {
      const since = awaySinceRef.current;
      if (since === null) {
        setAwayMs(0);
        setAlertActive(false);
        return;
      }
      const elapsed = Date.now() - since;

      // Debounce: a sub-second blip shouldn't visibly start the counter.
      const visibleElapsed = elapsed < AWAY_DEBOUNCE_MS ? 0 : elapsed;
      setAwayMs(visibleElapsed);

      const thresholdMs = thresholdRef.current * 1000;
      if (visibleElapsed >= thresholdMs) {
        setAlertActive(true);
        if (!notifiedRef.current) {
          notifiedRef.current = true;
          const state = lastGazeRef.current;
          void fireAwayNotification(state, Math.round(visibleElapsed / 1000));
        }
      } else {
        setAlertActive(false);
      }
    }, COUNTER_TICK_MS);

    return () => clearTicker();
  }, [tracking, clearTicker]);

  const handleStart = async () => {
    if (tracking) return;
    safeCleanup();
    setTracking(true);
    setStatus(
      "Starting Python tracker… If this sits here, check Python is on PATH (py, python) and ultralytics/OpenCV install.",
    );
    setFrame(null);

    void ensureNotificationPermission();

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
          handleFrame(f);
          setFrame((prev) => ({
            ...f,
            preview_jpeg_b64: f.preview_jpeg_b64 ?? prev?.preview_jpeg_b64,
          }));
          setStatus("Live — gaze tracking active");
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
  const gaze: GazeState = frame?.gaze_state ?? "absent";
  const debug = frame?.gaze_debug;
  const awaySec = Math.floor(awayMs / 1000);

  const pitchHot =
    debug?.pitch_ratio != null && debug.pitch_ratio < debug.down_threshold;
  const horizHot =
    debug?.horizontal_ratio != null &&
    (debug.horizontal_ratio < debug.left_threshold ||
      debug.horizontal_ratio > debug.right_threshold);
  const irisHot =
    debug?.iris_pitch != null && debug.iris_pitch > debug.iris_threshold;
  const pitchSource = debug?.pitch_source ?? null;

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>LockIN</h1>
          <p className="subtitle">
            Detects whether you're facing the monitor and nudges you when you drift.
          </p>
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
          <label className="field">
            <span>Alert after (s)</span>
            <input
              type="number"
              min={2}
              max={300}
              value={thresholdSec}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (Number.isFinite(v)) setThresholdSec(Math.max(2, Math.min(300, v)));
              }}
            />
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

      {alertActive && (
        <div className="alert-banner" role="alert">
          <strong>Eyes up.</strong> {GAZE_LABELS[gaze]} for {awaySec}s — refocus on the
          monitor.
        </div>
      )}

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
            <h2>Focus</h2>
            <div className="focus-row">
              <div className={`gaze-pill gaze-${gaze}`}>{GAZE_LABELS[gaze]}</div>
              {pitchSource ? (
                <span className={`source-tag source-${pitchSource}`}>
                  via {pitchSource}
                </span>
              ) : null}
            </div>
            <dl className="dl">
              <div>
                <dt>Away for</dt>
                <dd>{awayMs > 0 ? `${awaySec}s` : "—"}</dd>
              </div>
              <div>
                <dt>Threshold</dt>
                <dd>{thresholdSec}s</dd>
              </div>
            </dl>
          </div>

          <div className="metric">
            <h2>Eye movement</h2>
            <div
              className={`direction-pill dir-${(em?.direction ?? "steady").replace(/[^a-z0-9-]/gi, "")}`}
            >
              {em?.direction ?? "—"}
            </div>
            <dl className="dl">
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
              <div>
                <dt>Person visible</dt>
                <dd>{frame?.person.present ? "Yes" : "No"}</dd>
              </div>
            </dl>
          </div>

          <div className="metric">
            <h2>Gaze debug</h2>
            <dl className="dl">
              <div>
                <dt>Iris pitch</dt>
                <dd className={irisHot ? "hot" : ""}>
                  {debug?.iris_pitch != null
                    ? debug.iris_pitch.toFixed(3)
                    : "—"}
                  {debug ? (
                    <span className="threshold">
                      {" "}
                      &gt; {debug.iris_threshold.toFixed(2)} → down
                    </span>
                  ) : null}
                </dd>
              </div>
              <div>
                <dt>Iris L / R</dt>
                <dd>
                  {debug?.iris_left != null
                    ? debug.iris_left.toFixed(2)
                    : "—"}
                  {" / "}
                  {debug?.iris_right != null
                    ? debug.iris_right.toFixed(2)
                    : "—"}
                </dd>
              </div>
              <div>
                <dt>EAR L / R</dt>
                <dd>
                  {debug?.ear_left != null
                    ? debug.ear_left.toFixed(2)
                    : "—"}
                  {" / "}
                  {debug?.ear_right != null
                    ? debug.ear_right.toFixed(2)
                    : "—"}
                  {debug ? (
                    <span className="threshold">
                      {" "}
                      &lt; {debug.ear_min.toFixed(2)} → eye closed
                    </span>
                  ) : null}
                </dd>
              </div>
              <div>
                <dt>Head pitch</dt>
                <dd className={pitchHot ? "hot" : ""}>
                  {debug?.pitch_ratio != null
                    ? debug.pitch_ratio.toFixed(3)
                    : "—"}
                  {debug ? (
                    <span className="threshold">
                      {" "}
                      &lt; {debug.down_threshold.toFixed(2)} → down (fallback)
                    </span>
                  ) : null}
                </dd>
              </div>
              <div>
                <dt>Horiz. ratio</dt>
                <dd className={horizHot ? "hot" : ""}>
                  {debug?.horizontal_ratio != null
                    ? debug.horizontal_ratio.toFixed(3)
                    : "—"}
                  {debug ? (
                    <span className="threshold">
                      {" "}
                      &lt; {debug.left_threshold.toFixed(2)} or &gt;{" "}
                      {debug.right_threshold.toFixed(2)} → away
                    </span>
                  ) : null}
                </dd>
              </div>
              <div>
                <dt>Eyes visible</dt>
                <dd>{debug?.eyes_visible ?? "—"}</dd>
              </div>
            </dl>
          </div>
        </aside>
      </div>
    </div>
  );
}
