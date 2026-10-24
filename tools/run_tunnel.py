import os
import re
import sys
import time
import subprocess
import threading

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOUDFLARED_EXE = os.path.join(PROJECT_DIR, "tools", "cloudflared.exe")
VENV_PYTHON = os.path.join(PROJECT_DIR, ".venv", "Scripts", "python.exe")

if not os.path.exists(CLOUDFLARED_EXE):
    print(">> cloudflared.exe chưa có sẵn, đang tự động tải về...")
    try:
        import urllib.request
        os.makedirs(os.path.dirname(CLOUDFLARED_EXE), exist_ok=True)
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        urllib.request.urlretrieve(url, CLOUDFLARED_EXE)
        print(">> Tải cloudflared.exe thành công!")
    except Exception as e:
        print(f"[ERROR] Không thể tự động tải cloudflared.exe: {e}")
        sys.exit(1)

if not os.path.exists(VENV_PYTHON):
    print(f"[ERROR] Python venv not found at {VENV_PYTHON}")
    sys.exit(1)

def print_hud_banner(tunnel_url):
    print("\n" + "=" * 76)
    print("   NEURAL//VERIFY - AI RECOGNITION SERVER ONLINE (PUBLIC TUNNEL ACTIVATED)")
    print("=" * 76)
    print(f" [PUBLIC HTTPS / WSS DOMAIN]:")
    print(f"   -> {tunnel_url}")
    print()
    print(" [CÁCH SỬ DỤNG NHẬN DIỆN]:")
    print(f" 1. Cách trực tiếp (Khuyên dùng - Full tính năng):")
    print(f"    Mở trình duyệt bất kỳ (PC / Điện thoại 4G / Mạng ngoài):")
    print(f"    {tunnel_url}")
    print()
    print(" 2. Cách kết nối qua Render (https://vuongquochoa-detect-face.onrender.com):")
    clean_host = tunnel_url.replace("https://", "").replace("http://", "").strip("/")
    print(f"    - Vào Render: https://vuongquochoa-detect-face.onrender.com/")
    print(f"    - Bấm nút: '⚙ CẤU HÌNH MÁY CHỦ AI'")
    print(f"    - Nhập Host: {clean_host}")
    print(f"    - Bấm 'LƯU & KẾT NỐI' -> Hệ thống AI lập tức hoạt động!")
    print("=" * 76)
    print(" Nhấn Ctrl+C để dừng server.\n")

def main():
    print(">> Đang khởi động FastAPI Uvicorn Server trên cổng 8000...")
    server_process = subprocess.Popen(
        [VENV_PYTHON, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    tunnel_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    print(">> Đang thiết lập Cloudflare Tunnel mở cổng bảo mật HTTPS/WSS...")
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
        for line in tunnel_process.stdout:
            match = url_pattern.search(line)
            if match and not tunnel_url:
                tunnel_url = match.group(0)
                print_hud_banner(tunnel_url)

    tunnel_thread = threading.Thread(target=monitor_tunnel, daemon=True)
    tunnel_thread.start()

    try:
        while True:
            time.sleep(1)
            if server_process.poll() is not None:
                print("[ERROR] Uvicorn server đã dừng đột ngột!")
                break
            if tunnel_process.poll() is not None:
                print("[ERROR] Cloudflare tunnel đã dừng đột ngột!")
                break
    except KeyboardInterrupt:
        print("\n>> Đang dừng server và đóng tunnel...")
    finally:
        try:
            tunnel_process.terminate()
            server_process.terminate()
        except Exception:
            pass
        print(">> Đã đóng dịch vụ an toàn.")

if __name__ == "__main__":
    main()
