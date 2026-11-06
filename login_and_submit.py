import subprocess
import pty
import os
import sys
import json
import glob
import time
import urllib.request
import re

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

def do_login():
    print("\nStarting GitHub Device Flow login...")
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

            if "signed in as" in output.lower() or "token saved" in output.lower():
                logged_in = True
                break
        except OSError:
            break

    proc.wait()
    os.close(master)

    if logged_in or os.path.exists(USER_CONFIG_FILE):
        print("\n✓ Authentication verified!")
        if os.path.exists(USER_CONFIG_FILE):
            with open(USER_CONFIG_FILE, "r") as src, open(BACKUP_CONFIG, "w") as dst:
                dst.write(src.read())
            print("✓ Credentials backed up to /content/okc_config_backup.json")
        return True
    else:
        print("\nLogin flow did not complete. Please check the output above.", file=sys.stderr)
        return False

# 3. Check login status: config file must exist
if os.path.exists(USER_CONFIG_FILE) and os.path.getsize(USER_CONFIG_FILE) > 5:
    print("✓ Found existing OKC authentication token!")
else:
    if not do_login():
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

def run_submit():
    print("\n[4/4] Submitting to Meta H200 cluster & leaderboard...\n")
    sub_proc = subprocess.Popen(
        [OKC_BIN, "submit", "muse_bf16_proj/submission.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    submit_output = ""
    for line in sub_proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        submit_output += line
    sub_proc.wait()
    return submit_output

submit_output = run_submit()

# If not signed in, prompt login and retry submit
if "not signed in" in submit_output.lower():
    print("\n[!] Token was missing or invalid. Launching login now...")
    if do_login():
        submit_output = run_submit()

# If submission succeeded or matched an earlier job, auto-fetch status
match = re.search(r"okc status ([0-9a-fA-F]+)", submit_output)
if match:
    job_id = match.group(1)
    print(f"\n=== FETCHING DETAILED STATUS FOR JOB {job_id} ===")
    status_proc = subprocess.run([OKC_BIN, "status", job_id], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(status_proc.stdout)

# Also show leaderboard
print("\n=== CURRENT OKC LEADERBOARD ===")
lb_proc = subprocess.run([OKC_BIN, "leaderboard", "muse_bf16_proj"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(lb_proc.stdout)

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
