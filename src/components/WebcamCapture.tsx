import Webcam from 'react-webcam';

const videoConstraints = {
  width: 1280,
  height: 720,
  facingMode: 'user',
};

function WebcamCapture() {
  return (
   <Webcam videoConstraints={videoConstraints}></Webcam>
  );
}

export default WebcamCapture;