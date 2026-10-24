import os
import re
import sys
import time
import subprocess
import threading

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOUDFLARED_EXE = os.path.join(PROJECT_DIR, "tools", "cloudflared.exe")
VENV_PYTHON = os.path.join(PROJECT_DIR, ".venv", "Scripts", "python.exe")
TUNNEL_URL_FILE = os.path.join(PROJECT_DIR, "data", "tunnel_url.txt")

if not os.path.exists(CLOUDFLARED_EXE):
    print(">> cloudflared.exe chưa có sẵn, đang tự động tải về...", flush=True)
    try:
        import urllib.request
        os.makedirs(os.path.dirname(CLOUDFLARED_EXE), exist_ok=True)
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        urllib.request.urlretrieve(url, CLOUDFLARED_EXE)
        print(">> Tải cloudflared.exe thành công!", flush=True)
    except Exception as e:
        print(f"[ERROR] Không thể tự động tải cloudflared.exe: {e}", flush=True)
        sys.exit(1)

if not os.path.exists(VENV_PYTHON):
    print(f"[ERROR] Python venv not found at {VENV_PYTHON}", flush=True)
    sys.exit(1)

def print_hud_banner(tunnel_url):
    clean_host = tunnel_url.replace("https://", "").replace("http://", "").strip("/")
    os.makedirs(os.path.dirname(TUNNEL_URL_FILE), exist_ok=True)
    with open(TUNNEL_URL_FILE, "w", encoding="utf-8") as f:
        f.write(tunnel_url)

    banner = f"""
==============================================================================
   NEURAL//VERIFY - AI RECOGNITION SERVER ONLINE (PUBLIC TUNNEL ACTIVATED)
==============================================================================
 [1. LINK TRỰC TIẾP TỐI ƯU 0ms (Full SSL HTTPS / WSS)]:
    -> {tunnel_url}

 [2. LINK RENDER TỰ ĐỘNG CẤU HÌNH (Tự kết nối, không cần cài đặt)]:
    -> https://vuongquochoa-detect-face.onrender.com/?server={clean_host}

 [3. KẾT NỐI THỦ CÔNG (Nếu vào Render trực tiếp)]:
    - Mở https://vuongquochoa-detect-face.onrender.com/
    - Bấm nút: "🌐 SERVER: ..."
    - Nhập Host: {clean_host}
    - Bấm "LƯU & KẾT NỐI"
==============================================================================
 Nhấn Ctrl+C để dừng server.
"""
    print(banner, flush=True)

def main():
    print(">> Đang khởi động FastAPI Uvicorn Server trên cổng 8000...", flush=True)
    server_process = subprocess.Popen(
        [VENV_PYTHON, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"],
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    def drain_server_output():
        try:
            for line in server_process.stdout:
                if "ERROR" in line or "CRITICAL" in line:
                    print(f"[SERVER] {line.strip()}", flush=True)
        except Exception:
            pass

    server_thread = threading.Thread(target=drain_server_output, daemon=True)
    server_thread.start()

    tunnel_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    print(">> Đang thiết lập Cloudflare Tunnel mở cổng bảo mật HTTPS/WSS...", flush=True)
    tunnel_process = subprocess.Popen(
        [CLOUDFLARED_EXE, "tunnel", "--url", "http://127.0.0.1:8000"],
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    def monitor_tunnel():
        nonlocal tunnel_url
        try:
            for line in tunnel_process.stdout:
                match = url_pattern.search(line)
                if match and not tunnel_url:
                    tunnel_url = match.group(0)
                    print_hud_banner(tunnel_url)
        except Exception:
            pass

    tunnel_thread = threading.Thread(target=monitor_tunnel, daemon=True)
    tunnel_thread.start()

    try:
        while True:
            time.sleep(1)
            if server_process.poll() is not None:
                print("[ERROR] Uvicorn server đã dừng đột ngột!", flush=True)
                break
            if tunnel_process.poll() is not None:
                print("[ERROR] Cloudflare tunnel đã dừng đột ngột!", flush=True)
                break
    except KeyboardInterrupt:
        print("\n>> Đang dừng server và đóng tunnel...", flush=True)
    finally:
        try:
            tunnel_process.terminate()
            server_process.terminate()
        except Exception:
            pass
        print(">> Đã đóng dịch vụ an toàn.", flush=True)

if __name__ == "__main__":
    main()
