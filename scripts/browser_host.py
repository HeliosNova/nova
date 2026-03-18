#!/usr/bin/env python3
"""Host-side browser — launches a visible Chromium that Nova controls via CDP.

Run this on your desktop (not inside Docker). Nova connects over CDP.

Usage:
    python scripts/browser_host.py

Then add to .env:
    BROWSER_CDP_URL=http://host.docker.internal:9222

Nova will control this visible browser window in real-time.
"""

import glob
import os
import pathlib
import subprocess
import sys


def find_chromium() -> str:
    """Find Playwright's Chromium binary."""
    home = pathlib.Path.home()
    # Playwright on Windows
    pattern = str(home / "AppData/Local/ms-playwright/chromium-*/chrome-win64/chrome.exe")
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[-1]
    # Playwright on Windows (older layout)
    pattern = str(home / "AppData/Local/ms-playwright/chromium-*/chrome-win/chrome.exe")
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[-1]
    # Playwright on Linux
    pattern = str(home / ".cache/ms-playwright/chromium-*/chrome-linux/chrome")
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[-1]
    # macOS
    pattern = str(home / "Library/Caches/ms-playwright/chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium")
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[-1]
    return ""


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9222

    chrome = find_chromium()
    if not chrome:
        print("ERROR: Chromium not found. Run: python -m playwright install chromium")
        sys.exit(1)

    print(f"\n  Launching: {chrome}")
    print(f"  CDP port:  {port}")
    print(f"\n  Add to .env:  BROWSER_CDP_URL=http://host.docker.internal:{port}")
    print(f"\n  Close the browser window or press Ctrl+C to stop.\n")

    # Use separate user-data-dir so this doesn't interfere with regular Chrome
    import tempfile
    user_data = os.path.join(tempfile.gettempdir(), "nova-browser-profile")
    os.makedirs(user_data, exist_ok=True)

    args = [
        chrome,
        f"--remote-debugging-port={port}",
        "--remote-debugging-address=0.0.0.0",
        "--remote-allow-origins=*",
        f"--user-data-dir={user_data}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-infobars",
        "--start-maximized",
        "about:blank",
    ]

    try:
        proc = subprocess.run(args)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
