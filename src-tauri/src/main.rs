#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

use tauri::{AppHandle, Emitter, State};

struct TrackerChild(Arc<Mutex<Option<Child>>>);

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn spawn_python_script(script: &PathBuf, extra_args: &[String]) -> Result<Child, String> {
    let mut cmd = if cfg!(windows) {
        let mut c = Command::new("py");
        c.args(["-3", script.to_str().ok_or("invalid script path")?]);
        c
    } else {
        let mut c = Command::new("python3");
        c.arg(script);
        c
    };
    cmd.args(extra_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    cmd.spawn()
        .or_else(|_| {
            let mut c = Command::new("python");
            c.arg(script).args(extra_args);
            c.stdout(Stdio::piped()).stderr(Stdio::piped());
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                const CREATE_NO_WINDOW: u32 = 0x08000000;
                c.creation_flags(CREATE_NO_WINDOW);
            }
            c.spawn()
        })
        .map_err(|e| format!("failed to start Python ({})", e))
}

fn stop_tracker_inner(state: &TrackerChild) {
    if let Some(mut c) = state.0.lock().unwrap().take() {
        let _ = c.kill();
        let _ = c.wait();
    }
}

#[tauri::command]
fn list_cameras() -> Result<String, String> {
    let script = manifest_dir().join("binaries").join("list_cameras.py");
    if !script.is_file() {
        return Err(format!("missing {}", script.display()));
    }
    let out = if cfg!(windows) {
        Command::new("py")
            .args(["-3", script.to_str().unwrap()])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .or_else(|_| {
                Command::new("python")
                    .arg(&script)
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .output()
            })
    } else {
        Command::new("python3")
            .arg(&script)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
    }
    .map_err(|e| format!("list_cameras: {}", e))?;

    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[tauri::command]
fn start_tracking(
    app: AppHandle,
    camera_index: i32,
    child_state: State<'_, TrackerChild>,
) -> Result<(), String> {
    stop_tracker_inner(&child_state);

    let script: PathBuf = manifest_dir().join("binaries").join("eye_tracker.py");
    if !script.is_file() {
        return Err(format!("missing {}", script.display()));
    }

    let model_path = manifest_dir().join("models").join("yolo26n-pose.pt");
    let model_str = model_path.to_string_lossy().to_string();
    let extra = vec![model_str, camera_index.to_string()];

    let mut child = spawn_python_script(&script, &extra)?;

    if let Some(stderr) = child.stderr.take() {
        let app_err = app.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().flatten() {
                let _ = app_err.emit("tracker-error", line);
            }
        });
    }

    let stdout = child.stdout.take().ok_or("tracker has no stdout")?;
    {
        let mut g = child_state.0.lock().unwrap();
        *g = Some(child);
    }

    let app_out = app.clone();
    let arc = child_state.0.clone();
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(l) if !l.trim().is_empty() => {
                    let _ = app_out.emit("tracking-frame", l);
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
        {
            let mut g = arc.lock().unwrap();
            if let Some(mut c) = g.take() {
                let _ = c.wait();
            }
        }
        let _ = app_out.emit("tracker-stopped", ());
    });

    Ok(())
}

#[tauri::command]
fn stop_tracking(child_state: State<'_, TrackerChild>) -> Result<(), String> {
    stop_tracker_inner(&child_state);
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(TrackerChild(Arc::new(Mutex::new(None))))
        .invoke_handler(tauri::generate_handler![list_cameras, start_tracking, stop_tracking])
        .run(tauri::generate_context!())
        .expect("error running tauri app");
}
