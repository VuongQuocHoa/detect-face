"""
Automatic UDP Discovery Server & Beacon for FaceID CyberHUD.
Allows mobile devices on LAN to automatically locate the PC server without manual IP configuration,
and immediately wakes up clients via SERVER_ONLINE UDP Beacon within 0.1s.
"""

import json
import logging
import socket
import threading
import time

logger = logging.getLogger("discovery")

DISCOVERY_PORT = 8005
MAGIC_REQUEST = "FACEID_DISCOVER_REQUEST"
MAGIC_RESPONSE_HEADER = "FACEID_DISCOVER_RESPONSE"


def get_lan_ip() -> str:
    """Detect current active LAN IP of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class DiscoveryServer:
    def __init__(self, http_port: int = 8000, discovery_port: int = DISCOVERY_PORT):
        self.http_port = http_port
        self.discovery_port = discovery_port
        self.running = False
        self._thread = None
        self._beacon_thread = None
        self._sock = None

    def start(self):
        if self.running:
            return
        self.running = True

        self._thread = threading.Thread(target=self._listen_loop, daemon=True, name="UDP-DiscoveryListener")
        self._thread.start()

        self._beacon_thread = threading.Thread(target=self._beacon_loop, daemon=True, name="UDP-DiscoveryBeacon")
        self._beacon_thread.start()

        # Immediately blast SERVER_ONLINE burst across LAN to wake up hot-standby phones
        self.broadcast_server_online()

        print(f"[*] Auto-Discovery UDP Service started on port {self.discovery_port} (LAN IP: {get_lan_ip()}:{self.http_port})")

    def stop(self):
        self.running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass

    def broadcast_server_online(self):
        """Send an immediate rapid burst of SERVER_ONLINE packets to wake up connected clients within 0.1s."""
        def _burst():
            lan_ip = get_lan_ip()
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            payload = json.dumps({
                "header": "SERVER_ONLINE",
                "event": "SERVER_ONLINE",
                "server": "FaceID_CyberHUD",
                "ip": lan_ip,
                "port": self.http_port,
                "host": f"{lan_ip}:{self.http_port}",
                "timestamp": time.time()
            }).encode("utf-8")

            # Target standard global broadcast and all active interface subnet broadcasts
            targets = ["255.255.255.255"]
            try:
                import psutil, struct
                for iface, addrs in psutil.net_if_addrs().items():
                    for a in addrs:
                        if a.family == socket.AF_INET and not a.address.startswith("127."):
                            if a.broadcast:
                                targets.append(a.broadcast)
                            elif a.netmask:
                                ip_i = struct.unpack("!I", socket.inet_aton(a.address))[0]
                                mask_i = struct.unpack("!I", socket.inet_aton(a.netmask))[0]
                                targets.append(socket.inet_ntoa(struct.pack("!I", ip_i | (~mask_i & 0xFFFFFFFF))))
            except Exception:
                pass
            targets = list(set(targets))

            # Send rapid bursts 80ms apart across all broadcast targets to guarantee reception
            for _ in range(5):
                for target_ip in targets:
                    try:
                        sock.sendto(payload, (target_ip, self.discovery_port))
                    except Exception:
                        pass
                time.sleep(0.08)
            sock.close()
            print(f"[BEACON] Broadcasted SERVER_ONLINE burst for {lan_ip}:{self.http_port} to {targets}")

        threading.Thread(target=_burst, daemon=True, name="UDP-ServerOnlineBurst").start()

    def _listen_loop(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self._sock.bind(("0.0.0.0", self.discovery_port))
        except Exception as e:
            logger.error(f"[DiscoveryServer] Failed to bind UDP port {self.discovery_port}: {e}")
            return

        while self.running:
            try:
                data, addr = self._sock.recvfrom(2048)
                msg = data.decode("utf-8", errors="ignore").strip()
                if MAGIC_REQUEST in msg or "DISCOVER" in msg.upper() or "STATUS" in msg.upper():
                    lan_ip = get_lan_ip()
                    resp_dict = {
                        "header": MAGIC_RESPONSE_HEADER,
                        "event": "SERVER_ONLINE",
                        "server": "FaceID_CyberHUD",
                        "ip": lan_ip,
                        "port": self.http_port,
                        "host": f"{lan_ip}:{self.http_port}",
                        "timestamp": time.time()
                    }
                    resp_bytes = json.dumps(resp_dict).encode("utf-8")
                    self._sock.sendto(resp_bytes, addr)
                    print(f"[AUTO-DISCOVERY] Responded to mobile client at {addr[0]}:{addr[1]} -> Host: {lan_ip}:{self.http_port}")
            except Exception:
                if not self.running:
                    break
                time.sleep(0.1)

    def _beacon_loop(self):
        """Periodically broadcast SERVER_ONLINE presence on LAN so mobile devices can listen passively."""
        beacon_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        beacon_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        while self.running:
            try:
                lan_ip = get_lan_ip()
                beacon_dict = {
                    "header": "SERVER_ONLINE",
                    "event": "SERVER_ONLINE",
                    "server": "FaceID_CyberHUD",
                    "ip": lan_ip,
                    "port": self.http_port,
                    "host": f"{lan_ip}:{self.http_port}",
                    "timestamp": time.time()
                }
                msg = json.dumps(beacon_dict).encode("utf-8")
                beacon_sock.sendto(msg, ("255.255.255.255", self.discovery_port))
            except Exception:
                pass
            time.sleep(2.5)

        beacon_sock.close()


_global_discovery_server = None

def start_discovery_server(http_port: int = 8000, discovery_port: int = DISCOVERY_PORT):
    global _global_discovery_server
    if _global_discovery_server is None:
        _global_discovery_server = DiscoveryServer(http_port, discovery_port)
        _global_discovery_server.start()
    return _global_discovery_server
