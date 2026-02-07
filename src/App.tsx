import "./App.css";
import { useEffect } from "react";
import WebcamCapture from "./components/WebcamCapture";
import { invoke } from "@tauri-apps/api/core";
import { INIT_WEBCAM } from "./constants/RustHandler";

function App() {
  // useEffect(() => {
  //   invoke(INIT_WEBCAM).catch((err) => {
  //     console.error("Failed to initialize webcam:", err);
  //   });
  // });

  return (
   <WebcamCapture />
  );
}

export default App;
