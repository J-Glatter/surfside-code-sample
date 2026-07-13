# Windows 3080 box — remote render worker setup

End state: the box sits powered off; from the Mac (or anywhere via Tailscale)
you wake it, drop job files into a shared folder, and collect finished sprites.
`spriteforge worker` runs automatically at boot and keeps the SD model warm
between jobs. Sections 1–2 are one-time Windows configuration; 3–5 install the
software; 6 is the daily flow.

## 1. Remote power-on (Wake-on-LAN)

**BIOS/UEFI** (one time): enable *Wake on LAN / Resume by PCI-E device*
(vendors name it differently — often under Power Management or Advanced →
APM). While there, set *Restore AC Power Loss → Power On* if you want the
smart-plug fallback below.

**Windows** (one time, admin PowerShell):

```powershell
# Fast Startup's hybrid shutdown breaks WoL from full power-off on many boards
powercfg /h off

# NIC settings — or Device Manager -> your Ethernet adapter -> Power Management:
#   [x] Allow this device to wake the computer
#   [x] Only allow a magic packet to wake the computer
# and Advanced tab: "Wake on Magic Packet" = Enabled
Get-NetAdapter | Format-Table Name, MacAddress   # note the MAC address

# Give the box a fixed IP or a DHCP reservation in the router while you're at it
```

Use **Ethernet** — WoL over Wi-Fi is unreliable-to-nonexistent.

**Waking it from the Mac** (same LAN):

```bash
brew install wakeonlan
wakeonlan AA:BB:CC:DD:EE:FF        # the MAC from above
```

**Waking it from outside the LAN:** a magic packet must originate *inside* the
LAN, so you need one always-on device there. Any of: a Raspberry Pi / spare
router with SSH (`ssh pi wakeonlan <MAC>` over Tailscale), a router with a
built-in WoL page, or the blunt fallback — a smart plug + the BIOS
*power-on-after-power-loss* setting, so cutting and restoring power boots the
box. Test the full off → wake → SSH loop before relying on it.

## 2. Remote access

**OpenSSH server** (built into Windows, one time, admin PowerShell):

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service sshd -StartupType Automatic
Start-Service sshd

# Key auth. Gotcha: for users in the Administrators group, Windows reads keys
# from a shared file, NOT ~/.ssh/authorized_keys:
#   C:\ProgramData\ssh\administrators_authorized_keys
# Create it, paste your Mac's public key in, then fix its ACL:
icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r `
  /grant "Administrators:F" /grant "SYSTEM:F"

# Make PowerShell the default shell for ssh sessions
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
  -Value "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -PropertyType String -Force
```

**Tailscale** (recommended): install on the Windows box, the Mac, and the
always-on WoL sender. Everything below then works identically from anywhere,
no port forwarding, and SMB/SSH/RDP all ride the tailnet.

**RDP**: enable (Settings → System → Remote Desktop) for GUI needs — kohya's
web UI, eyeballing folders. Day-to-day triggering shouldn't need it.

## 3. Software environment

Admin PowerShell:

```powershell
winget install Git.Git Python.Python.3.11

git clone https://github.com/J-Glatter/surfside-code-sample spriteforge
cd spriteforge

python -m venv .venv
.venv\Scripts\Activate.ps1

# CUDA torch FIRST (plain `pip install torch` grabs a wheel that may lack CUDA
# on Windows). Check pytorch.org "Get Started" for the current index URL:
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -e ".[generate,curate,animate,director]"

# Credentials for the LLM director (optional — heuristics work without it)
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-...", "User")
```

Smoke test (downloads ~4 GB of SD weights on first run, then confirm the CUDA
path — it should NOT print the CPU fallback warning):

```powershell
spriteforge make "a small slime monster" -o smoke
```

That single command is Checkpoint A's first half — see `docs/CHECKPOINTS.md`.

## 4. The jobs share

```powershell
mkdir C:\sprite-jobs
# Share it (Explorer -> folder Properties -> Sharing), or:
New-SmbShare -Name jobs -Path C:\sprite-jobs -FullAccess "$env:USERNAME"
```

On the Mac: Finder → Go → Connect to Server → `smb://<box-ip-or-tailscale-name>/jobs`.
Anything you write there is on the box instantly; results appear in
`done/` the same way. (Alternative if you'd rather avoid SMB: Syncthing a
folder both ways.)

## 5. The worker, auto-started at boot

Test it interactively first:

```powershell
spriteforge worker C:\sprite-jobs --palette C:\spriteforge\game.json
```

Then register it as a scheduled task so it survives reboots and runs headless
(admin PowerShell — adjust paths):

```powershell
$action = New-ScheduledTaskAction -Execute "C:\spriteforge\.venv\Scripts\spriteforge.exe" `
  -Argument "worker C:\sprite-jobs --palette C:\spriteforge\game.json"
$trigger = New-ScheduledTaskTrigger -AtStartup
Register-ScheduledTask -TaskName "spriteforge-worker" -Action $action -Trigger $trigger `
  -User "$env:USERNAME" -Password (Read-Host "password") -RunLevel Highest
```

Notes: CUDA compute works fine from a scheduled task (no display needed).
Set *Settings → Power* so the box never sleeps while on AC (`powercfg /change
standby-timeout-ac 0`), or the worker sleeps with it mid-queue.

## 6. Daily flow from the Mac

```bash
wakeonlan AA:BB:CC:DD:EE:FF                        # 1. power on (~30s to worker up)

spriteforge plan "a dire wolf" > /Volumes/jobs/wolf.json     # 2a. planned job
echo "mossy cobblestone ground" > /Volumes/jobs/cobble.txt   # 2b. or bare prompt

open /Volumes/jobs/done/wolf/                      # 3. collect results
touch /Volumes/jobs/STOP                           # 4. optional: stop the worker
ssh windows-box "shutdown /s /t 60"                # 5. optional: power down
```

Job files: a `.json` is a plan exactly as `spriteforge plan` prints it (add a
top-level `"seed": 7` to vary generation); a `.txt` is a bare prompt the
director plans on arrival. Failures land in `failed/<name>/error.txt` instead
of crashing the worker.

## 7. kohya_ss (LoRA training — Checkpoint C)

```powershell
git clone https://github.com/bmaltais/kohya_ss
cd kohya_ss; .\setup.bat        # its own venv + GUI; run via .\gui.bat, use over RDP
```

`spriteforge dataset prep` output (folder layout + `kohya_config.toml` +
`NOTES.md`) plugs straight into it.

## 8. Troubleshooting

| Symptom | Check |
|---|---|
| WoL does nothing | Fast Startup back on after a Windows update (`powercfg /h off` again); NIC "Wake on Magic Packet" reset by a driver update; packet sent from outside the LAN |
| Worker generates on CPU (slow, prints warning) | torch installed without CUDA — reinstall from the cu-index URL; verify `python -c "import torch; print(torch.cuda.is_available())"` |
| SSH key auth refused for admin user | key must be in `administrators_authorized_keys` with the strict ACL, not `~/.ssh/authorized_keys` |
| First job takes minutes | expected once per boot: model load + (first ever run) the ~4 GB download; subsequent jobs are seconds — that's why the worker keeps the pipe warm |
| Jobs stuck in the inbox | worker not running: `Get-ScheduledTask spriteforge-worker`; check power/sleep settings |
