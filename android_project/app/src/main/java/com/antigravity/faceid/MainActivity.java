package com.antigravity.faceid;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.net.DhcpInfo;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.webkit.JavascriptInterface;
import android.webkit.PermissionRequest;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.InetSocketAddress;

public class MainActivity extends Activity {

    private static final String TAG = "FaceID_CyberHUD";
    private static final int PERMISSION_REQ_CODE = 1001;
    private static final int DISCOVERY_PORT = 8005;

    private WebView webView;
    private String lastDiscoveredHost = null;
    private boolean isDiscovering = false;
    private long lastDiscoveryTime = 0;

    private DatagramSocket beaconListenerSocket = null;
    private WifiManager.MulticastLock persistentMulticastLock = null;
    private volatile boolean isRunning = true;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Immersive Fullscreen Cyberpunk Display
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(
                WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN
        );
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        webView = new WebView(this);
        setContentView(webView);

        configureWebView();

        // Start UDP Auto-Discovery request on startup
        startAutoDiscovery();

        // Start persistent UDP SERVER_ONLINE Beacon Listener for 0.1s hot recovery
        startBeaconListener();

        // Check & Request Camera & Microphone Permissions
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            String[] perms = new String[]{Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO};
            boolean needPerm = false;
            for (String p : perms) {
                if (checkSelfPermission(p) != PackageManager.PERMISSION_GRANTED) {
                    needPerm = true;
                    break;
                }
            }
            if (needPerm) {
                requestPermissions(perms, PERMISSION_REQ_CODE);
            } else {
                loadWebApp();
            }
        } else {
            loadWebApp();
        }
    }

    private void configureWebView() {
        WebSettings ws = webView.getSettings();
        ws.setJavaScriptEnabled(true);
        ws.setDomStorageEnabled(true);
        ws.setDatabaseEnabled(true);
        ws.setMediaPlaybackRequiresUserGesture(false);
        ws.setAllowFileAccess(true);
        ws.setAllowContentAccess(true);
        ws.setAllowFileAccessFromFileURLs(true);
        ws.setAllowUniversalAccessFromFileURLs(true);
        ws.setLoadWithOverviewMode(true);
        ws.setUseWideViewPort(true);

        // Hardware-accelerated Canvas & WebGL
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            webView.setLayerType(View.LAYER_TYPE_HARDWARE, null);
        }

        // Allow Mixed Content (http/ws within assets)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            ws.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);
        }

        // Bridge for JavaScript <-> Native Android Communication
        webView.addJavascriptInterface(new WebAppInterface(), "AndroidBridge");

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageFinished(WebView view, String url) {
                super.onPageFinished(view, url);
                if (lastDiscoveredHost != null) {
                    notifyJavaScript(lastDiscoveredHost);
                }
            }
        });

        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(final PermissionRequest request) {
                MainActivity.this.runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                            request.grant(request.getResources());
                        }
                    }
                });
            }
        });
    }

    public class WebAppInterface {
        @JavascriptInterface
        public void triggerAutoDiscovery() {
            startAutoDiscovery();
        }

        @JavascriptInterface
        public String getDiscoveredHost() {
            return lastDiscoveredHost != null ? lastDiscoveredHost : "";
        }
    }

    private void loadWebApp() {
        webView.loadUrl("file:///android_asset/index.html");
    }

    /**
     * Listens continuously for UDP Beacon "SERVER_ONLINE" on LAN.
     * When server starts, this intercepts packet within milliseconds and reconnects in 0.1s.
     */
    private void startBeaconListener() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    WifiManager wifi = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
                    if (wifi != null) {
                        persistentMulticastLock = wifi.createMulticastLock("FaceIDBeaconLock");
                        persistentMulticastLock.setReferenceCounted(true);
                        persistentMulticastLock.acquire();
                    }

                    beaconListenerSocket = new DatagramSocket(null);
                    beaconListenerSocket.setReuseAddress(true);
                    beaconListenerSocket.setBroadcast(true);
                    beaconListenerSocket.bind(new InetSocketAddress(DISCOVERY_PORT));

                    byte[] buf = new byte[2048];
                    DatagramPacket packet = new DatagramPacket(buf, buf.length);

                    while (isRunning) {
                        try {
                            beaconListenerSocket.receive(packet);
                            String msg = new String(packet.getData(), 0, packet.getLength(), "UTF-8").trim();
                            if (msg.contains("SERVER_ONLINE") || msg.contains("FACEID_BEACON") || msg.contains("FACEID_DISCOVER_RESPONSE")) {
                                JSONObject json = new JSONObject(msg);
                                final String serverHost = json.optString("host", "");
                                if (!serverHost.isEmpty()) {
                                    lastDiscoveredHost = serverHost;
                                    Log.i(TAG, "⚡ Intercepted SERVER_ONLINE Beacon: " + serverHost);
                                    runOnUiThread(new Runnable() {
                                        @Override
                                        public void run() {
                                            notifyServerOnline(serverHost);
                                        }
                                    });
                                }
                            }
                        } catch (Exception e) {
                            if (!isRunning) break;
                        }
                    }
                } catch (Exception e) {
                    Log.w(TAG, "Beacon listener error: " + e.getMessage());
                }
            }
        }, "UDP-BeaconListener").start();
    }

    private void notifyServerOnline(final String host) {
        if (webView == null || host == null) return;
        final String js = "if (window.onServerOnline) { window.onServerOnline('" + host + "'); } " +
                          "else if (window.onServerDiscovered) { window.onServerDiscovered('" + host + "'); } " +
                          "else { localStorage.setItem('ai_server_host', '" + host + "'); }";
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            webView.evaluateJavascript(js, null);
        } else {
            webView.loadUrl("javascript:" + js);
        }
    }

    /**
     * Sends UDP Broadcast discovery packet to locate PC server on LAN automatically.
     */
    public synchronized void startAutoDiscovery() {
        long now = System.currentTimeMillis();
        if (isDiscovering || (now - lastDiscoveryTime < 10000)) return;
        lastDiscoveryTime = now;
        isDiscovering = true;

        new Thread(new Runnable() {
            @Override
            public void run() {
                WifiManager.MulticastLock lock = null;
                DatagramSocket socket = null;
                try {
                    WifiManager wifi = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
                    if (wifi != null) {
                        lock = wifi.createMulticastLock("FaceIDMulticastLock");
                        lock.setReferenceCounted(true);
                        lock.acquire();
                    }

                    socket = new DatagramSocket();
                    socket.setBroadcast(true);
                    socket.setSoTimeout(2500);

                    byte[] reqData = "FACEID_DISCOVER_REQUEST".getBytes("UTF-8");

                    // 1. Send to standard global broadcast
                    try {
                        DatagramPacket p1 = new DatagramPacket(reqData, reqData.length, InetAddress.getByName("255.255.255.255"), DISCOVERY_PORT);
                        socket.send(p1);
                    } catch (Exception ignored) {}

                    // 2. Send to subnet directed broadcast (e.g., 10.0.97.255)
                    try {
                        InetAddress subnetBcast = getSubnetBroadcastAddress(wifi);
                        if (subnetBcast != null) {
                            DatagramPacket p2 = new DatagramPacket(reqData, reqData.length, subnetBcast, DISCOVERY_PORT);
                            socket.send(p2);
                        }
                    } catch (Exception ignored) {}

                    // 3. Listen for server response
                    byte[] buf = new byte[2048];
                    DatagramPacket respPacket = new DatagramPacket(buf, buf.length);
                    socket.receive(respPacket);

                    String respStr = new String(respPacket.getData(), 0, respPacket.getLength(), "UTF-8");
                    JSONObject json = new JSONObject(respStr);
                    final String serverHost = json.optString("host", "");

                    if (!serverHost.isEmpty()) {
                        if (serverHost.equals(lastDiscoveredHost)) {
                            return; // Already discovered, ignore duplicate
                        }
                        lastDiscoveredHost = serverHost;
                        Log.i(TAG, "⚡ Server auto-discovered: " + serverHost);

                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                notifyJavaScript(serverHost);
                            }
                        });
                    }
                } catch (Exception e) {
                    Log.w(TAG, "Auto-discovery attempt finished: " + e.getMessage());
                } finally {
                    isDiscovering = false;
                    if (socket != null && !socket.isClosed()) {
                        socket.close();
                    }
                    if (lock != null && lock.isHeld()) {
                        try {
                            lock.release();
                        } catch (Exception ignored) {}
                    }
                }
            }
        }).start();
    }

    private void notifyJavaScript(final String host) {
        if (webView == null || host == null) return;
        final String js = "if (window.onServerDiscovered) { window.onServerDiscovered('" + host + "'); } " +
                          "else { localStorage.setItem('ai_server_host', '" + host + "'); }";
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            webView.evaluateJavascript(js, null);
        } else {
            webView.loadUrl("javascript:" + js);
        }
    }

    private InetAddress getSubnetBroadcastAddress(WifiManager wifi) throws IOException {
        if (wifi == null) return null;
        DhcpInfo dhcp = wifi.getDhcpInfo();
        if (dhcp == null || dhcp.ipAddress == 0) return null;
        int broadcast = (dhcp.ipAddress & dhcp.netmask) | ~dhcp.netmask;
        byte[] quads = new byte[4];
        for (int k = 0; k < 4; k++) {
            quads[k] = (byte) ((broadcast >> (k * 8)) & 0xFF);
        }
        return InetAddress.getByAddress(quads);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQ_CODE) {
            boolean allGranted = true;
            for (int res : grantResults) {
                if (res != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }
            if (allGranted) {
                loadWebApp();
            } else {
                Toast.makeText(this, "Vui lòng cấp quyền Camera trong Cài đặt ứng dụng!", Toast.LENGTH_LONG).show();
                loadWebApp();
            }
        }
    }

    @Override
    public void onBackPressed() {
        if (webView != null && webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }

    @Override
    protected void onDestroy() {
        isRunning = false;
        if (beaconListenerSocket != null && !beaconListenerSocket.isClosed()) {
            beaconListenerSocket.close();
        }
        if (persistentMulticastLock != null && persistentMulticastLock.isHeld()) {
            try {
                persistentMulticastLock.release();
            } catch (Exception ignored) {}
        }
        super.onDestroy();
    }
}
