use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType, ApiBackend};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::Camera;
use image::{ImageBuffer, Rgb, DynamicImage};
use std::thread;

#[tauri::command]
pub fn start_camera() {
    let index = CameraIndex::Index(0);
    let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = Camera::with_backend(index, requested, ApiBackend::MediaFoundation).expect("Could not open camera");
    camera.open_stream().expect("Could not start stream");
}

