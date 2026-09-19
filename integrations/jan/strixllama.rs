//! strixllama management over native IPC; the helper never exposes a network port.
use serde_json::Value;
use std::{io::Write, path::PathBuf, process::{Command, Stdio}};

#[tauri::command]
pub async fn strixllama_request(request: Value) -> Result<Value, String> {
    let op = request.get("op").and_then(Value::as_str).ok_or("Missing operation")?;
    if !["catalog", "status", "logs", "profile", "roots", "save", "start", "stop"].contains(&op) {
        return Err("Unsupported strixllama operation".into());
    }
    let input = serde_json::to_vec(&request).map_err(|e| e.to_string())?;
    if input.len() > 65536 { return Err("Request exceeds 64 KiB".into()); }
    tauri::async_runtime::spawn_blocking(move || {
        let root = std::env::var_os("STRIX_ROOT").map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3).unwrap().to_path_buf()
        });
        // STRIX_PYTHON if set, otherwise whatever `python` resolves to on PATH. It used to default
        // to one machine's embedded interpreter, which no other machine has.
        let python = std::env::var_os("STRIX_PYTHON").map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("python"));
        let helper = root.join("tools/manager.py");
        if !helper.is_file() {
            return Err(format!("找不到 strixllama 管理器 {}，可用 STRIX_ROOT 指定仓库根目录", helper.display()));
        }
        let mut command = Command::new(python);
        command.arg(helper).current_dir(root).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        #[cfg(windows)] {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        let mut child = command.spawn().map_err(|e| format!(
            "无法启动 Python（{}）：{e}。请确认 python 在 PATH 中，或用 STRIX_PYTHON 指定解释器",
            python.display()))?;
        child.stdin.take().ok_or("Missing helper stdin")?.write_all(&input).map_err(|e| e.to_string())?;
        let output = child.wait_with_output().map_err(|e| e.to_string())?;
        let result: Value = serde_json::from_slice(&output.stdout).map_err(|e| format!("管理器响应无效: {e}; {}", String::from_utf8_lossy(&output.stderr)))?;
        if result.get("ok").and_then(Value::as_bool) != Some(true) {
            return Err(result.get("error").and_then(Value::as_str).unwrap_or("管理操作失败").to_owned());
        }
        Ok(result["data"].clone())
    }).await.map_err(|e| e.to_string())?
}
