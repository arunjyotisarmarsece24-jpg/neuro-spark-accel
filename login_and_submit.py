import subprocess
import pty
import os
import sys
import json
import glob
import time
import urllib.request

OKC_BIN = "/root/.okc/bin/okc"
BACKUP_CONFIG = "/content/okc_config_backup.json"
USER_CONFIG_DIR = "/root/.config/okc"
USER_CONFIG_FILE = os.path.join(USER_CONFIG_DIR, "config.json")

# 1. Ensure OKC is installed
if not os.path.exists(OKC_BIN):
    print("Installing okc CLI...")
    res = subprocess.run("curl -fsSL https://openkernelchallenge.dev/install.sh | sh", shell=True)
    if res.returncode != 0:
        print("Failed to install okc CLI!", file=sys.stderr)
        sys.exit(1)

# 2. Restore token from backup if available
if not os.path.exists(USER_CONFIG_FILE) and os.path.exists(BACKUP_CONFIG):
    print("Restoring saved credentials from /content/okc_config_backup.json...")
    os.makedirs(USER_CONFIG_DIR, exist_ok=True)
    try:
        with open(BACKUP_CONFIG, "r") as src, open(USER_CONFIG_FILE, "w") as dst:
            dst.write(src.read())
    except Exception as e:
        print("Backup restore note:", e)

# 3. Check login status via `okc problems`
check = subprocess.run([OKC_BIN, "problems"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
if check.returncode == 0:
    print("✓ Already authenticated with OKC cluster!")
    logged_in = True
else:
    print("\nStarting automated GitHub Device Flow login...")
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        [OKC_BIN, "login"],
        stdin=slave,
        stdout=master,
        stderr=master,
        close_fds=True
    )
    os.close(slave)

    output = ""
    logged_in = False
    consent_sent = False

    while True:
        try:
            chunk = os.read(master, 1024).decode('utf-8', errors='ignore')
            if not chunk:
                break
            output += chunk
            sys.stdout.write(chunk)
            sys.stdout.flush()

            # Auto-accept terms
            if "Accept terms and conditions?" in output and not consent_sent:
                os.write(master, b"y\n")
                consent_sent = True

            # Auto-answer recruiter disclosure
            if "recruiter" in output.lower() and "[y/N]" in output:
                os.write(master, b"n\n")

            if "signed in as" in output.lower():
                logged_in = True
                break
        except OSError:
            break

    proc.wait()
    os.close(master)

    if logged_in or os.path.exists(USER_CONFIG_FILE):
        print("\n✓ Authentication verified!")
        logged_in = True
        if os.path.exists(USER_CONFIG_FILE):
            with open(USER_CONFIG_FILE, "r") as src, open(BACKUP_CONFIG, "w") as dst:
                dst.write(src.read())
            print("✓ Credentials backed up to /content/okc_config_backup.json")
    else:
        print("Login flow did not complete. Please check the output above.", file=sys.stderr)
        sys.exit(1)

# 4. Fetch the latest Beast Kernel (cache-busted)
print("\n[3/4] Downloading latest Beast Kernel from GitHub...")
os.makedirs("muse_bf16_proj", exist_ok=True)
kernel_url = f"https://raw.githubusercontent.com/arunjyotisarmarsece24-jpg/neuro-spark-accel/main/colab_submission.py?v={int(time.time())}"
req = urllib.request.Request(kernel_url, headers={"Cache-Control": "no-cache"})
with urllib.request.urlopen(req) as resp, open("muse_bf16_proj/submission.py", "wb") as f_out:
    f_out.write(resp.read())
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

# 6. Parse and print cluster tune and disable log
logs = sorted(glob.glob("muse_bf16_proj/profiles/*.log"), key=os.path.getmtime, reverse=True)
if logs:
    try:
        with open(logs[0], "r", encoding="utf-8", errors="replace") as f:
            log_text = f.read()
        print("\n=== CLUSTER TUNE & DISABLE LOG ===")
        for line in log_text.splitlines():
            if "[okc]" in line or "disabled:" in line or "error" in line.lower():
                print(line)
    except Exception as e:
        print("Log read error:", e)
