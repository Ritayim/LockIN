import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export interface Camera {
  index: number;
  label: string;
  width: number;
  height: number;
  fps: number;
}

export interface EyeMovement {
  dx: number;
  dy: number;
  magnitude: number;
  direction: string;
  has_movement: boolean;
}

export interface Point2D {
  x: number;
  y: number;
}

export interface EyeState {
  right: Point2D;
  left: Point2D;
  right_delta: Point2D;
  left_delta: Point2D;
}

export interface HeadState {
  position: Point2D;
  delta: Point2D;
  yaw: number;
  pitch: number;
}

export interface PersonState {
  present: boolean;
  count: number;
  bbox: { x: number; y: number; width: number; height: number } | null;
}

export type GazeState = "straight" | "down" | "away" | "absent";

export interface GazeDebug {
  pitch_ratio: number | null;
  horizontal_ratio: number | null;
  eyes_visible: "both" | "left" | "right" | "none";
  iris_pitch: number | null;
  iris_left: number | null;
  iris_right: number | null;
  ear_left: number | null;
  ear_right: number | null;
  pitch_source: "eye" | "head" | null;
  down_threshold: number;
  iris_threshold: number;
  ear_min: number;
  left_threshold: number;
  right_threshold: number;
}

export interface TrackingFrame {
  person: PersonState;
  head: HeadState | null;
  eyes: EyeState | null;
  eye_movement: EyeMovement;
  gaze_state: GazeState;
  gaze_debug?: GazeDebug;
  timestamp_ms: number;
  preview_jpeg_b64?: string;
}

export async function getCameras(): Promise<Camera[]> {
  const raw = await invoke<string>("list_cameras");
  return JSON.parse(raw) as Camera[];
}

export async function stopTracking(): Promise<void> {
  await invoke("stop_tracking");
}

/** Subscribe to tracker events; call `cleanup()` when stopping or unmounting. */
export async function subscribeTracking(
  onFrame: (frame: TrackingFrame) => void,
  onError?: (msg: string) => void,
  onStopped?: () => void,
): Promise<{ cleanup: () => void }> {
  const unsubs: UnlistenFn[] = [];

  unsubs.push(
    await listen<string>("tracking-frame", ({ payload }) => {
      try {
        const data = JSON.parse(payload) as Record<string, unknown>;
        if (typeof data.eye_movement !== "object" || data.eye_movement === null) {
          return;
        }
        onFrame(data as unknown as TrackingFrame);
      } catch (e) {
        console.error("tracking-frame parse", e);
      }
    }),
  );

  unsubs.push(
    await listen<string>("tracker-error", ({ payload }) => {
      onError?.(payload);
    }),
  );

  unsubs.push(
    await listen("tracker-stopped", () => {
      onStopped?.();
    }),
  );

  return {
    cleanup: () => {
      for (const u of unsubs) u();
    },
  };
}

export async function startTrackingSession(
  cameraIndex: number,
  onFrame: (frame: TrackingFrame) => void,
  onError?: (msg: string) => void,
  onStopped?: () => void,
): Promise<{ cleanup: () => void }> {
  const { cleanup } = await subscribeTracking(onFrame, onError, onStopped);
  await invoke("start_tracking", { cameraIndex });
  return {
    cleanup: () => {
      cleanup();
    },
  };
}
