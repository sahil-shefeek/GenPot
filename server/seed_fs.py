"""
server/seed_fs.py

Seeds ``logs/world_state.json`` with a realistic Ubuntu 22.04 filesystem so
the honeypot looks convincing from the very first attacker connection.

Usage
-----
    uv run python -m server.seed_fs          # from project root
    uv run python server/seed_fs.py          # direct

Design
------
* Loads the existing state file first so attacker-written data is never lost.
* Adds each default entry **only if the path is not already present** in the
  VFS — idempotent, safe to run multiple times.
* Content is intentionally realistic and enticing: a leaked ``/etc/shadow``,
  a ``/root/.bash_history`` that hints at database credentials, and
  ``/var/log/auth.log`` full of brute-force attempts that make the machine
  look like a juicy live target.
"""
from __future__ import annotations

import os
import sys

# Allow running as ``python server/seed_fs.py`` from the project root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.state_manager import StateManager  # noqa: E402

# ── Default filesystem entries ────────────────────────────────────────────────
# Each entry: (absolute_path, metadata_dict)
# ``content`` uses realistic Ubuntu 22.04 text.

_DEFAULTS: list[tuple[str, dict]] = [
    # ── /etc ──────────────────────────────────────────────────────────────────
    ("/etc", {"type": "directory", "owner": "root", "permissions": "755"}),
    (
        "/etc/hostname",
        {
            "type": "file",
            "owner": "root",
            "permissions": "644",
            "content": "genpot\n",
        },
    ),
    (
        "/etc/hosts",
        {
            "type": "file",
            "owner": "root",
            "permissions": "644",
            "content": (
                "127.0.0.1\tlocalhost\n"
                "127.0.1.1\tgenpot\n"
                "::1\t\tlocalhost ip6-localhost ip6-loopback\n"
                "ff02::1\t\tip6-allnodes\n"
                "ff02::2\t\tip6-allrouters\n"
            ),
        },
    ),
    (
        "/etc/os-release",
        {
            "type": "file",
            "owner": "root",
            "permissions": "644",
            "content": (
                'PRETTY_NAME="Ubuntu 22.04.3 LTS"\n'
                'NAME="Ubuntu"\n'
                'VERSION_ID="22.04"\n'
                'VERSION="22.04.3 LTS (Jammy Jellyfish)"\n'
                'VERSION_CODENAME=jammy\n'
                "ID=ubuntu\n"
                "ID_LIKE=debian\n"
                'HOME_URL="https://www.ubuntu.com/"\n'
                'SUPPORT_URL="https://help.ubuntu.com/"\n'
                'BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"\n'
            ),
        },
    ),
    (
        "/etc/issue",
        {
            "type": "file",
            "owner": "root",
            "permissions": "644",
            "content": "Ubuntu 22.04.3 LTS \\n \\l\n\n",
        },
    ),
    (
        "/etc/passwd",
        {
            "type": "file",
            "owner": "root",
            "permissions": "644",
            "content": (
                "root:x:0:0:root:/root:/bin/bash\n"
                "daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin\n"
                "bin:x:2:2:bin:/bin:/usr/sbin/nologin\n"
                "sys:x:3:3:sys:/dev:/usr/sbin/nologin\n"
                "sync:x:4:65534:sync:/bin:/bin/sync\n"
                "games:x:5:60:games:/usr/games:/usr/sbin/nologin\n"
                "man:x:6:12:man:/var/cache/man:/usr/sbin/nologin\n"
                "lp:x:7:7:lp:/var/spool/lpd:/usr/sbin/nologin\n"
                "mail:x:8:8:mail:/var/mail:/usr/sbin/nologin\n"
                "news:x:9:9:news:/var/spool/news:/usr/sbin/nologin\n"
                "uucp:x:10:10:uucp:/var/spool/uucp:/usr/sbin/nologin\n"
                "proxy:x:13:13:proxy:/bin:/usr/sbin/nologin\n"
                "www-data:x:33:33:www-data:/var/www:/usr/sbin/nologin\n"
                "backup:x:34:34:backup:/var/backups:/usr/sbin/nologin\n"
                "nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n"
                "_apt:x:100:65534::/nonexistent:/usr/sbin/nologin\n"
                "systemd-network:x:101:102:systemd Network Management,,,:/run/systemd:/usr/sbin/nologin\n"
                "systemd-resolve:x:102:103:systemd Resolver,,,:/run/systemd:/usr/sbin/nologin\n"
                "messagebus:x:103:104::/nonexistent:/usr/sbin/nologin\n"
                "sshd:x:106:65534::/run/sshd:/usr/sbin/nologin\n"
                "syslog:x:107:113::/home/syslog:/usr/sbin/nologin\n"
                "ubuntu:x:1000:1000:Ubuntu:/home/ubuntu:/bin/bash\n"
                "deploy:x:1001:1001:Deploy User:/home/deploy:/bin/bash\n"
            ),
        },
    ),
    (
        "/etc/shadow",
        {
            "type": "file",
            "owner": "root",
            "permissions": "640",
            "content": (
                "root:$6$rounds=656000$rAnd0mSalt123456$GQxTXxD3WQa8L7p2kZvM"
                "n9Y4hXsPwJFdK1LmNbVcUeRqWoItHgEsAyZjCfBu5OiDl6Tk0Nv.:19358:0:99999:7:::\n"
                "daemon:*:19000:0:99999:7:::\n"
                "bin:*:19000:0:99999:7:::\n"
                "sys:*:19000:0:99999:7:::\n"
                "sync:*:19000:0:99999:7:::\n"
                "nobody:*:19000:0:99999:7:::\n"
                "sshd:!:19358::::::\n"
                "ubuntu:$6$rounds=656000$uBuntuSalt789012$MxWqPyLz2VaK8n3jH6sR"
                "e1FbGtYcXdNmOpQrUwJiCvA4EgBfZkDl5Th7So9Iu0Pj.:19360:0:99999:7:::\n"
                "deploy:$6$rounds=656000$deployS4lt345678$KpRqNmVx9WaL2bJ5zT8y"
                "A3hGcFdUeXoMnPsQwIjBvC6ElDkYr1Sg4Fu7Ho0It.:19365:0:99999:7:::\n"
            ),
        },
    ),
    (
        "/etc/group",
        {
            "type": "file",
            "owner": "root",
            "permissions": "644",
            "content": (
                "root:x:0:\n"
                "daemon:x:1:\n"
                "bin:x:2:\n"
                "sys:x:3:\n"
                "adm:x:4:syslog,ubuntu\n"
                "tty:x:5:\n"
                "disk:x:6:\n"
                "lp:x:7:\n"
                "mail:x:8:\n"
                "news:x:9:\n"
                "uucp:x:10:\n"
                "man:x:12:\n"
                "proxy:x:13:\n"
                "www-data:x:33:\n"
                "backup:x:34:\n"
                "nobody:x:65534:\n"
                "sudo:x:27:ubuntu\n"
                "docker:x:998:ubuntu,deploy\n"
                "ubuntu:x:1000:\n"
                "deploy:x:1001:\n"
            ),
        },
    ),
    # ── /etc/ssh ──────────────────────────────────────────────────────────────
    ("/etc/ssh", {"type": "directory", "owner": "root", "permissions": "755"}),
    (
        "/etc/ssh/sshd_config",
        {
            "type": "file",
            "owner": "root",
            "permissions": "644",
            "content": (
                "# This is the sshd server system-wide configuration file.\n"
                "# See sshd_config(5) for more information.\n\n"
                "Port 22\n"
                "AddressFamily any\n"
                "ListenAddress 0.0.0.0\n\n"
                "HostKey /etc/ssh/ssh_host_rsa_key\n"
                "HostKey /etc/ssh/ssh_host_ecdsa_key\n"
                "HostKey /etc/ssh/ssh_host_ed25519_key\n\n"
                "SyslogFacility AUTH\n"
                "LogLevel INFO\n\n"
                "LoginGraceTime 2m\n"
                "PermitRootLogin yes\n"
                "StrictModes yes\n"
                "MaxAuthTries 6\n"
                "MaxSessions 10\n\n"
                "PubkeyAuthentication yes\n"
                "AuthorizedKeysFile .ssh/authorized_keys\n\n"
                "PasswordAuthentication yes\n"
                "PermitEmptyPasswords no\n\n"
                "UsePAM yes\n"
                "X11Forwarding yes\n"
                "PrintMotd no\n"
                "AcceptEnv LANG LC_*\n"
                "Subsystem sftp /usr/lib/openssh/sftp-server\n"
            ),
        },
    ),
    # ── /root ─────────────────────────────────────────────────────────────────
    ("/root", {"type": "directory", "owner": "root", "permissions": "700"}),
    (
        "/root/.bashrc",
        {
            "type": "file",
            "owner": "root",
            "permissions": "644",
            "content": (
                "# ~/.bashrc: executed by bash(1) for non-login shells.\n\n"
                "export HISTSIZE=10000\n"
                "export HISTFILESIZE=20000\n"
                "export HISTCONTROL=ignoreboth\n\n"
                "alias ll='ls -alF'\n"
                "alias la='ls -A'\n"
                "alias l='ls -CF'\n\n"
                "export DB_HOST=localhost\n"
                "export DB_PORT=5432\n"
                "export DB_NAME=appdb\n"
                "export DB_USER=appuser\n"
                "# DB_PASS exported in .bash_profile (not here)\n"
            ),
        },
    ),
    (
        "/root/.bash_history",
        {
            "type": "file",
            "owner": "root",
            "permissions": "600",
            "content": (
                "apt update && apt upgrade -y\n"
                "systemctl status nginx\n"
                "systemctl restart nginx\n"
                "cat /etc/nginx/nginx.conf\n"
                "cd /var/www/html\n"
                "ls -la\n"
                "vim index.html\n"
                "cd /root\n"
                "mysql -u root -p'S3cur3DB!2024'\n"
                "mysqldump -u root -p appdb > /root/backup_2024.sql\n"
                "ls -la /root\n"
                "cat /root/.ssh/authorized_keys\n"
                "vim /etc/ssh/sshd_config\n"
                "systemctl restart sshd\n"
                "docker ps -a\n"
                "docker logs webapp_1\n"
                "cd /opt/deploy\n"
                "cat .env\n"
                "git pull origin main\n"
                "history\n"
            ),
        },
    ),
    (
        "/root/.ssh",
        {"type": "directory", "owner": "root", "permissions": "700"},
    ),
    (
        "/root/.ssh/authorized_keys",
        {
            "type": "file",
            "owner": "root",
            "permissions": "600",
            "content": (
                "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC3mP+8zRXvK2lN9wJqD"
                "mH4tYbGfL7sEuVcAiW0pKxOeRnZdQyT6hFvBjMsN1LkP3rXgCwD5oIaE"
                "2VuZfNpQsSbR admin@workstation\n"
            ),
        },
    ),
    # ── /var/log ──────────────────────────────────────────────────────────────
    ("/var", {"type": "directory", "owner": "root", "permissions": "755"}),
    ("/var/log", {"type": "directory", "owner": "root", "permissions": "755"}),
    (
        "/var/log/syslog",
        {
            "type": "file",
            "owner": "syslog",
            "permissions": "640",
            "content": (
                "Mar  2 04:00:01 genpot CRON[2341]: (root) CMD (test -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.daily ))\n"
                "Mar  2 04:17:01 genpot CRON[2398]: (root) CMD (   cd / && run-parts --report /etc/cron.hourly)\n"
                "Mar  2 05:45:22 genpot kernel: [12045.338271] audit: type=1400 audit(1677729922.311:52): apparmor=\"ALLOWED\"\n"
                "Mar  2 06:00:01 genpot CRON[2512]: (root) CMD (test -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.daily ))\n"
                "Mar  2 07:17:01 genpot CRON[2641]: (root) CMD (   cd / && run-parts --report /etc/cron.hourly)\n"
                "Mar  2 08:00:01 genpot CRON[2789]: (root) CMD (test -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.daily ))\n"
                "Mar  2 09:15:44 genpot systemd[1]: nginx.service: Reloading.\n"
                "Mar  2 09:15:44 genpot systemd[1]: Reloaded A high performance web server.\n"
                "Mar  2 10:00:01 genpot CRON[3001]: (root) CMD (test -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.daily ))\n"
                "Mar  2 12:00:01 genpot CRON[3245]: (root) CMD (test -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.daily ))\n"
            ),
        },
    ),
    (
        "/var/log/auth.log",
        {
            "type": "file",
            "owner": "syslog",
            "permissions": "640",
            "content": (
                "Mar  2 01:12:03 genpot sshd[4521]: Failed password for root from 45.83.64.1 port 51234 ssh2\n"
                "Mar  2 01:12:05 genpot sshd[4521]: Failed password for root from 45.83.64.1 port 51234 ssh2\n"
                "Mar  2 01:12:07 genpot sshd[4521]: Failed password for root from 45.83.64.1 port 51234 ssh2\n"
                "Mar  2 01:14:19 genpot sshd[4589]: Failed password for admin from 194.165.16.77 port 33210 ssh2\n"
                "Mar  2 01:14:21 genpot sshd[4589]: Failed password for admin from 194.165.16.77 port 33210 ssh2\n"
                "Mar  2 02:33:44 genpot sshd[5103]: Invalid user oracle from 185.220.101.45 port 41099\n"
                "Mar  2 02:33:44 genpot sshd[5103]: Failed password for invalid user oracle from 185.220.101.45 port 41099 ssh2\n"
                "Mar  2 03:05:12 genpot sshd[5412]: Accepted password for root from 203.0.113.42 port 54321 ssh2\n"
                "Mar  2 03:05:12 genpot sshd[5412]: pam_unix(sshd:session): session opened for user root by (uid=0)\n"
                "Mar  2 04:23:11 genpot sshd[6001]: Accepted password for root from 203.0.113.42 port 54322 ssh2\n"
                "Mar  2 04:23:11 genpot sshd[6001]: pam_unix(sshd:session): session opened for user root by (uid=0)\n"
                "Mar  2 08:41:55 genpot sshd[7234]: Failed password for ubuntu from 91.92.251.103 port 48231 ssh2\n"
                "Mar  2 08:41:57 genpot sshd[7234]: Failed password for ubuntu from 91.92.251.103 port 48231 ssh2\n"
                "Mar  2 10:19:03 genpot sshd[8012]: Failed password for root from 198.199.122.45 port 55001 ssh2\n"
                "Mar  2 10:19:05 genpot sshd[8012]: Failed password for root from 198.199.122.45 port 55001 ssh2\n"
            ),
        },
    ),
    # ── /opt/deploy ───────────────────────────────────────────────────────────
    ("/opt", {"type": "directory", "owner": "root", "permissions": "755"}),
    ("/opt/deploy", {"type": "directory", "owner": "deploy", "permissions": "755"}),
    (
        "/opt/deploy/.env",
        {
            "type": "file",
            "owner": "deploy",
            "permissions": "600",
            "content": (
                "APP_ENV=production\n"
                "APP_SECRET_KEY=xK9#mP2$vL8nQ4wR\n"
                "DB_HOST=localhost\n"
                "DB_PORT=5432\n"
                "DB_NAME=appdb\n"
                "DB_USER=appuser\n"
                "DB_PASS=S3cur3DB!2024\n"
                "REDIS_URL=redis://localhost:6379/0\n"
                "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
                "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n"
                "AWS_DEFAULT_REGION=us-east-1\n"
                "JWT_SECRET=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9_secret\n"
            ),
        },
    ),
    # ── /proc ─────────────────────────────────────────────────────────────────
    ("/proc", {"type": "directory", "owner": "root", "permissions": "555"}),
    (
        "/proc/version",
        {
            "type": "file",
            "owner": "root",
            "permissions": "444",
            "content": (
                "Linux version 5.15.0-91-generic (buildd@lcy02-amd64-017) "
                "(gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0, GNU ld (GNU Binutils for Ubuntu) 2.38) "
                "#101-Ubuntu SMP Tue Nov 14 13:30:08 UTC 2023\n"
            ),
        },
    ),
    # ── /etc/cron ─────────────────────────────────────────────────────────────
    (
        "/etc/crontab",
        {
            "type": "file",
            "owner": "root",
            "permissions": "644",
            "content": (
                "# /etc/crontab: system-wide crontab\n"
                "SHELL=/bin/sh\n"
                "PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin\n\n"
                "17 *\t* * *\troot\tcd / && run-parts --report /etc/cron.hourly\n"
                "25 6\t* * *\troot\ttest -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.daily )\n"
                "47 6\t* * 7\troot\ttest -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.weekly )\n"
                "52 6\t1 * *\troot\ttest -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.monthly )\n"
                "0 2\t* * *\troot\t/opt/deploy/scripts/backup.sh >> /var/log/backup.log 2>&1\n"
            ),
        },
    ),
]


def seed(state_file: str = "logs/world_state.json") -> None:
    """
    Populate *state_file* with default Linux filesystem entries.

    Entries are only written if the path is not already present in the VFS,
    so attacker-written files are never overwritten.
    """
    sm = StateManager(state_file=state_file)
    fs = sm.state.get("filesystem", {})

    updates = []
    skipped = 0
    added = 0

    for path, meta in _DEFAULTS:
        if path in fs:
            skipped += 1
            continue
        updates.append({"action": "SET", "scope": "filesystem", "key": path, "value": meta})
        added += 1

    if updates:
        sm.apply_updates(updates)

    print(f"[seed_fs] Done — added {added} entries, skipped {skipped} already-existing entries.")
    print(f"[seed_fs] State file: {os.path.abspath(state_file)}")


if __name__ == "__main__":
    seed()
