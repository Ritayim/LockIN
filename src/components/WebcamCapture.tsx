import { useState } from 'react';
import Webcam from 'react-webcam';
import { invoke } from '@tauri-apps/api/core';

const videoConstraints = {
  width: 1280,
  height: 720,
  facingMode: 'user',
};

interface EyeMovement {
  dx: number;
  dy: number;
  direction: string;
  has_movement: boolean;
}

interface EyeDetectionResult {
  detections: unknown[];
  eye_movement: EyeMovement;
}
function WebcamCapture() {
  const [eyeResult, setEyeResult] = useState<EyeMovement | null>(null);
  const [loading, setLoading] = useState(false);

  const detectEyeMovement = async () => {
    setLoading(true);
    setEyeResult(null);
    try {
      const result = await invoke<EyeDetectionResult>('detect_eye_movement');
      setEyeResult(result.eye_movement);
    } catch (e) {
      console.error('Eye detection failed:', e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 16 }}>
      <Webcam videoConstraints={videoConstraints} />
      <div style={{ marginTop: 16 }}>
        <button
          type="button"
          onClick={detectEyeMovement}
          disabled={loading}
        >
          {loading ? 'Detecting…' : 'Detect eye movement'}
        </button>
        {eyeResult && (
          <p style={{ marginTop: 8 }}>
            Direction: <strong>{eyeResult.direction}</strong>
            {eyeResult.has_movement && (
              <> (Δx: {eyeResult.dx.toFixed(0)}, Δy: {eyeResult.dy.toFixed(0)})</>
            )}
          </p>
        )}
      </div>
    </div>
  );
}

export default WebcamCapture;
