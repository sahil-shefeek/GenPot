"""
Test script to simulate a live SMTP attack.

Connects to the local SMTP honeypot to verify it properly handles
the protocol handshake, data mode payload, and disconnect scenarios.
"""

import socket
import time


def test_smtp_attack(host: str = "127.0.0.1", port: int = 8025):
    """Run an interactive SMTP session over raw sockets to test the emulator."""

    print(f"Connecting to {host}:{port}...")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5.0)
            s.connect((host, port))

            # Read 220 Greeting
            greeting = s.recv(1024).decode("utf-8").strip()
            print(f"Server: {greeting}")

            commands = [
                b"EHLO attacker.com\r\n",
                b"MAIL FROM:<hacker@evil.com>\r\n",
                b"RCPT TO:<admin@corp.com>\r\n",
                b"DATA\r\n",
                b"Subject: Phishing\r\n\r\nClick this link to reset your password!\r\n.\r\n",
                b"QUIT\r\n",
            ]

            for cmd in commands:
                time.sleep(0.5)  # slight delay for realism
                print(f"\nClient: {cmd.decode('utf-8').strip()}")
                s.sendall(cmd)

                # We may receive multiple lines if there are multi-line responses
                response = s.recv(4096).decode("utf-8").strip()
                print(f"Server:\n{response}")

            print("\n*** Attack simulation complete ***")

    except ConnectionRefusedError:
        print(f"Error: Connection refused. Is the GenPot server running on {port}?")
    except socket.timeout:
        print("Error: Socket timed out waiting for response.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    test_smtp_attack()
