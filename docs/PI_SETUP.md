# Pi 4 midpoint — complete setup checklist

Topology this covers: **Pi 4 joins the house LAN over Wi-Fi; one Ethernet
cable runs directly Pi ↔ PC** (private `10.0.0.x` segment carrying wake
packets + the jobs share). The PC keeps its own Wi-Fi for internet. For the
simpler everything-wired-to-the-router topology, use `wakeonlan <MAC>` in
step 5 instead of etherwake and skip step 3.

## 0. Flash the card (on the Mac)

Raspberry Pi Imager → **Raspberry Pi OS Lite (64-bit)**. Before writing, open
the customisation settings (gear icon) and set:

- hostname: `pi`
- username `pi` + enable SSH with your public key
- your Wi-Fi SSID + password, and locale

Boot the Pi (Wi-Fi only is fine for setup), then from the Mac:

```bash
ssh pi@pi.local
```

## 1. Update everything, once

```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

## 2. Install the packages

```bash
sudo apt install -y samba etherwake wakeonlan inotify-tools
```

(`etherwake` for the direct link, `wakeonlan` kept as the LAN-topology spare,
`inotify-tools` for the watcher, `samba` for the share.)

## 3. Static IP on the PC link (direct-link topology only)

Bookworm uses NetworkManager — give eth0 a fixed address and **no gateway**,
so the default route stays on Wi-Fi:

```bash
sudo nmcli con add type ethernet ifname eth0 con-name pc-link \
  ipv4.method manual ipv4.addresses 10.0.0.1/24
sudo nmcli con up pc-link
```

Windows side (Settings → Network → Ethernet → IP assignment → Manual):
IP `10.0.0.2`, mask `255.255.255.0`, gateway and DNS **left empty** (its
Wi-Fi provides those). Verify from the Pi: `ping 10.0.0.2` while the PC is on.

## 4. The jobs share

```bash
sudo mkdir -p /srv/jobs && sudo chown pi:pi /srv/jobs

sudo tee -a /etc/samba/smb.conf > /dev/null <<'EOF'

[jobs]
   path = /srv/jobs
   writeable = yes
   valid users = pi
EOF

sudo smbpasswd -a pi        # sets the share password (can differ from login)
sudo systemctl restart smbd
```

Test: Mac → Finder → Go → Connect to Server → `smb://pi.local/jobs`;
PC → Explorer → `\\10.0.0.1\jobs` (then
`cmdkey /add:10.0.0.1 /user:pi /pass:...` so the worker's scheduled task can
connect unattended).

## 5. The wake-on-job watcher

Get the PC's **Ethernet** MAC first (PowerShell: `Get-NetAdapter`), then:

```bash
sudo tee /usr/local/bin/wake-on-job.sh > /dev/null <<'EOF'
#!/usr/bin/env bash
JOBS=/srv/jobs
PC_MAC=AA:BB:CC:DD:EE:FF     # <-- the PC's wired NIC
PC_IP=10.0.0.2
inotifywait -m -e create -e moved_to --format '%f' "$JOBS" | while read -r f; do
  case "$f" in
    *.json|*.txt)
      ping -c1 -W1 "$PC_IP" >/dev/null 2>&1 || /usr/sbin/etherwake -i eth0 "$PC_MAC" ;;
  esac
done
EOF
sudo chmod +x /usr/local/bin/wake-on-job.sh
```

(`etherwake` sends a raw frame out eth0 specifically — plain `wakeonlan`
would broadcast via the Wi-Fi default route, where the PC isn't listening.)

## 6. Run it as a service

```bash
sudo tee /etc/systemd/system/wake-on-job.service > /dev/null <<'EOF'
[Unit]
Description=Wake the render PC when a job lands
After=network-online.target smbd.service
Wants=network-online.target

[Service]
ExecStart=/usr/local/bin/wake-on-job.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now wake-on-job
systemctl status wake-on-job          # should be active (running)
```

## 7. Tailscale (remote access from anywhere)

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up                     # prints a login URL — open it once
```

Install Tailscale on the Mac too; `smb://pi/jobs` then works from anywhere.

## 8. Test the chain, in stages

```bash
# a) watcher fires (PC on): watch the log while dropping a file from the Mac
journalctl -u wake-on-job -f
# Mac: echo "a small slime monster" > /Volumes/jobs/slime.txt
#      -> log shows the event; no wake sent because the ping succeeded

# b) wake works: shut the PC down cleanly, drop another job file
#      -> PC powers on within seconds

# c) end to end: with the worker's scheduled task installed (WINDOWS_SETUP.md
#    §5, pointed at \\10.0.0.1\jobs), the same drop boots the PC, renders,
#    and results appear in /srv/jobs/done/slime/ a few minutes later
```

## Maintenance

Practically none. `sudo apt update && sudo apt full-upgrade -y` when you think
of it; `journalctl -u wake-on-job` if wakes ever stop; use a decent-brand SD
card — that's the only part of an always-on Pi that wears out.
