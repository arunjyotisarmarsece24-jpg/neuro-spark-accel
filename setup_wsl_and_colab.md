# Zero-to-One Linux & OKC Setup Guide (For Windows Users)

Since the `okc` CLI tool and GPU testing containers are designed for Linux, you have two frictionless paths. You **do not** need to reformat your machine or install dual-boot Linux.

---

## OPTION 1: Windows Subsystem for Linux (WSL 2)
*Best for: Developing locally, offline coding, editing in VS Code on Windows.*

### Step 1: Install WSL 2 & Ubuntu
1. On your Windows desktop, press `Win + X` and select **Terminal (Admin)** or **PowerShell (Admin)**.
2. Run:
   ```powershell
   wsl --install -d Ubuntu
   ```
3. When the download finishes, restart your Windows computer if prompted.
4. After restart, Ubuntu will open and ask you to create a **UNIX username** and **password** (choose anything simple, e.g., `arunjyoti`).

### Step 2: Access your Windows Project Files inside WSL
Your Windows files are automatically mounted inside WSL under `/mnt/c/`.  
To navigate straight to this contest folder inside WSL:
```bash
cd "/mnt/c/Users/IITBHU RESEARCH LAB/OneDrive/Desktop/Arunjyoti_Sarma/My_Coding_files/My_contest_Project/Muse Spark Open_Kernel_challenge"
```

### Step 3: Install `okc` CLI inside WSL
Run the official OKC installation command:
```bash
curl -fsSL https://openkernelchallenge.dev/install.sh | sh
```
Reload your shell path:
```bash
source ~/.bashrc
```

### Step 4: Login & Check Problems
```bash
# 1. Login with your GitHub account (opens browser / device code)
okc login

# 2. View all 5 challenge problems and your daily quotas
okc problems

# 3. Initialize your first problem workspace:
okc init muse_dense_projections
```

---

## OPTION 2: Google Colab (Zero-Install, Cloud Linux + Free GPU)
*Best for: Immediate start in 60 seconds with no local downloads, plus free access to a remote GPU for running and testing Triton code!*

### Step 1: Open Google Colab
1. Go to [https://colab.research.google.com](https://colab.research.google.com).
2. Click **New Notebook**.
3. Go to **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Select **T4 GPU** (or A100 if you have Colab Pro).

### Step 2: Install `okc` in Colab
In the first cell of your notebook, run:
```bash
!curl -fsSL https://openkernelchallenge.dev/install.sh | sh
!export PATH="/root/.okc/bin:$PATH"
```

### Step 3: Authenticate in Colab
```bash
!/root/.okc/bin/okc login
```
Follow the URL and enter the 8-character device code on GitHub.

### Step 4: Work and Submit Directly from Colab
```bash
# See problems:
!/root/.okc/bin/okc problems

# Init problem:
!/root/.okc/bin/okc init muse_dense_projections

# Test & Validate:
!/root/.okc/bin/okc check

# Submit to the official H200 evaluation cluster:
!/root/.okc/bin/okc submit submission.py
```

---

## OPTION 3: GitHub Codespaces (Browser-based Linux VS Code)
1. Go to your GitHub profile and create a private repository (e.g., `muse-open-kernel-challenge`).
2. Click the green `<> Code` button $\rightarrow$ **Codespaces** $\rightarrow$ **Create codespace on main**.
3. You get a full Linux terminal and VS Code inside your web browser.
4. Run `curl -fsSL https://openkernelchallenge.dev/install.sh | sh` and `okc login`.
