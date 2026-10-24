import glob
import os
import shutil
import subprocess
import zipfile

def main():
    print("=================================================================")
    print("   Building FaceID CyberHUD APK with Native UDP Auto-Discovery   ")
    print("=================================================================")

    src_java = "android_project/app/src/main/java/com/antigravity/faceid/MainActivity.java"
    src_manifest = "android_project/app/src/main/AndroidManifest.xml"
    src_assets = "android_project/app/src/main/assets"
    src_html = "app/templates/index.html"
    bin_dir = "android_project/app/src/main/bin"
    dex_dir = os.path.join(bin_dir, "dex")
    base_apk = os.path.join(bin_dir, "base.apk")
    compiled_res = os.path.join(bin_dir, "compiled_res.zip")
    dist_dir = "dist"
    final_apk = os.path.join(dist_dir, "FaceID_CyberHUD.apk")

    java_bin = r"C:\Program Files\JetBrains\PyCharm 2026.2.1\jbr\bin\java.exe"
    javac_bin = r"C:\Program Files\JetBrains\PyCharm 2026.2.1\jbr\bin\javac.exe"
    aapt2_bin = "tools/aapt2.exe"
    r8_jar = "tools/r8.jar"
    android_slim = "tools/android_slim.jar"
    android_all = "tools/android-all-11.jar"
    signer_jar = "tools/uber-apk-signer.jar"

    os.makedirs(src_assets, exist_ok=True)
    os.makedirs(bin_dir, exist_ok=True)
    os.makedirs(dex_dir, exist_ok=True)
    os.makedirs(dist_dir, exist_ok=True)

    # 1. Sync HTML to assets
    shutil.copy2(src_html, os.path.join(src_assets, "index.html"))
    print(f" [1/5] Synced index.html -> assets ({os.path.getsize(src_html)} bytes)")

    # 2. Compile Java
    print(" [2/5] Compiling MainActivity.java with javac (low-memory)...")
    res = subprocess.run([
        javac_bin, "-J-Xmx64m", "--release", "8",
        "-cp", android_slim,
        "-d", bin_dir,
        src_java
    ], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] javac failed:\n{res.stderr}\n{res.stdout}")
        return False

    # 3. Compile DEX with D8
    print(" [3/5] Converting bytecode to classes.dex with D8...")
    class_files = glob.glob(os.path.join(bin_dir, "com", "antigravity", "faceid", "*.class"))
    res = subprocess.run([
        java_bin, "-Xmx64m", "-cp", r8_jar,
        "com.android.tools.r8.D8",
        "--min-api", "21",
        "--lib", android_slim,
        "--output", dex_dir
    ] + class_files, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] D8 failed:\n{res.stderr}\n{res.stdout}")
        return False

    # 4. Link resources & assets into APK with AAPT2
    print(" [4/5] Packaging APK package with aapt2 link...")
    res = subprocess.run([
        aapt2_bin, "link",
        "-I", android_all,
        "--manifest", src_manifest,
        "-A", src_assets,
        "-o", base_apk,
        compiled_res,
        "--auto-add-overlay"
    ], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] aapt2 link failed:\n{res.stderr}\n{res.stdout}")
        return False

    # Inject classes.dex into base.apk
    dex_path = os.path.join(dex_dir, "classes.dex")
    with zipfile.ZipFile(base_apk, "a") as z:
        z.write(dex_path, "classes.dex")

    # 5. Sign APK
    print(" [5/5] Aligning & Signing APK with uber-apk-signer...")
    temp_signed = os.path.join(dist_dir, "base-aligned-debugSigned.apk")
    if os.path.exists(temp_signed):
        os.remove(temp_signed)
    if os.path.exists(final_apk):
        os.remove(final_apk)

    res = subprocess.run([
        java_bin, "-Xmx64m", "-jar", signer_jar,
        "-a", base_apk,
        "-o", dist_dir
    ], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] uber-apk-signer failed:\n{res.stderr}\n{res.stdout}")
        return False

    if os.path.exists(temp_signed):
        shutil.move(temp_signed, final_apk)
    
    if os.path.exists(final_apk):
        size_kb = os.path.getsize(final_apk) / 1024.0
        print(f"\n[SUCCESS] Built signed APK with UDP Auto-Discovery: {final_apk} ({size_kb:.1f} KB)")
        return True
    else:
        print("[ERROR] Final APK not found!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
