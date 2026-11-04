#!/usr/bin/env python3
import os
import pty
import select
import shutil
import subprocess
import sys
import time
import urllib.request

CONFIG_DIR = "/root/.config/okc"
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
BACKUP_FILE = "/content/okc_config_backup.json"
OKC_BIN = "/root/.okc/bin/okc"

print("=================================================================")
print("🚀 OKC LEADERBOARD RUNNER - NVIDIA H200 SM90a")
print("=================================================================")

# 1. Ensure okc binary is present
if not os.path.exists(OKC_BIN):
    print("\n[1/4] Installing okc CLI...")
    subprocess.run("curl -fsSL https://openkernelchallenge.dev/install.sh | sh", shell=True, check=True)
os.environ["PATH"] = "/root/.okc/bin:" + os.environ.get("PATH", "")

# 2. Restore token from backup if available
if not os.path.exists(CONFIG_FILE) and os.path.exists(BACKUP_FILE):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    shutil.copy(BACKUP_FILE, CONFIG_FILE)
    print("✓ Restored OKC auth token from backup!")

# 3. Check login status
needs_login = True
if os.path.exists(CONFIG_FILE):
    test_run = subprocess.run([OKC_BIN, "problems"], capture_output=True, text=True)
    if "not signed in" not in test_run.stdout.lower() and test_run.returncode == 0:
        needs_login = False
        print("\n[2/4] ✓ Already signed into OKC!")

if needs_login:
    print("\n[2/4] Authenticating with GitHub Device Flow...")
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        [OKC_BIN, "login"],
        stdin=slave, stdout=slave, stderr=slave, close_fds=True
    )
    os.close(slave)

    # Automatically send 'y\n' to agree to challenge terms
    time.sleep(1.0)
    os.write(master, b"y\n")

    authenticated = False
    start_time = time.time()
    while time.time() - start_time < 300:  # 5 min timeout
        r, _, _ = select.select([master], [], [], 0.5)
        if r:
            try:
                chunk = os.read(master, 1024)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="ignore")
                sys.stdout.write(text)
                sys.stdout.flush()

                if "recruiting and business purposes" in text:
                    # Automatically send 'y\n' for recruiting question
                    os.write(master, b"y\n")

                if "token saved" in text or "signed in as" in text:
                    authenticated = True
                    # Let it flush the rest of message
                    time.sleep(1.0)
                    break
            except OSError:
                break
        elif proc.poll() is not None:
            break

    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        pass

    # Backup the token
    if os.path.exists(CONFIG_FILE):
        os.makedirs(os.path.dirname(BACKUP_FILE), exist_ok=True)
        shutil.copy(CONFIG_FILE, BACKUP_FILE)
        print("\n✓ Auth token backed up successfully!")
    else:
        print("\n[!] Please make sure you authorized the code in your browser.")

# 4. Fetch the latest 199 KB Beast Kernel (commit c6faf91)
print("\n[3/4] Downloading latest 199 KB Beast Kernel (commit c6faf91)...")
os.makedirs("muse_bf16_proj", exist_ok=True)
kernel_url = "https://raw.githubusercontent.com/arunjyotisarmarsece24-jpg/neuro-spark-accel/main/colab_submission.py"
urllib.request.urlretrieve(kernel_url, "muse_bf16_proj/submission.py")
k_size = os.path.getsize("muse_bf16_proj/submission.py")
print(f"✓ Kernel ready: {k_size:,} bytes")

# 5. Submit directly to Meta H200 cluster
print("\n[4/4] Submitting to Meta H200 cluster & leaderboard...\n")
sub_proc = subprocess.Popen(
    [OKC_BIN, "submit", "muse_bf16_proj/submission.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)
for line in sub_proc.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
sub_proc.wait()
