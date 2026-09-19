"""
Root Launcher for Jindal Stainless Steel Defect Detection Dashboard.
Runs from either the root NEU-DET folder or steel_defect_detection folder.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Locate project directory
ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "steel_defect_detection" if (ROOT / "steel_defect_detection").exists() else ROOT
APP_PATH = PROJECT_DIR / "dashboard" / "app.py"

# Add Python user scripts to PATH
user_scripts = Path(os.path.expanduser("~")) / "AppData" / "Roaming" / "Python" / "Python314" / "Scripts"
if user_scripts.exists():
    os.environ["PATH"] = str(user_scripts) + os.pathsep + os.environ.get("PATH", "")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Jindal Defect Detection Dashboard for Deployment")
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0"),
        help="Server host address (default: 0.0.0.0 for deployment)",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=os.getenv("PORT", "8501"),
        help="Server port (default: 8501)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=os.getenv("STREAMLIT_SERVER_HEADLESS", "true").lower() in ("true", "1", "yes"),
        help="Run Streamlit in headless mode (default: True for deployment)",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        default=False,
        help="Open browser automatically (sets headless to false)",
    )
    args, unknown = parser.parse_known_args()

    headless_mode = "false" if args.browser else ("true" if args.headless else "false")

    print(f"============================================================")
    print(f"  Launching Jindal Stainless AI Defect Detection Platform   ")
    print(f"  Team: GenCoders (Lipi Bagarti & Kartik Ranjan Singh)       ")
    print(f"============================================================")
    print(f"Dashboard File: {APP_PATH}")
    print(f"Network URL:    http://{args.host}:{args.port}")
    print(f"Local URL:      http://localhost:{args.port}")
    print(f"Headless Mode:  {headless_mode}\n")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.address",
        str(args.host),
        "--server.port",
        str(args.port),
        "--server.headless",
        headless_mode,
        "--browser.gatherUsageStats",
        "false",
        *unknown,
    ]

    try:
        subprocess.run(cmd, cwd=str(PROJECT_DIR))
    except KeyboardInterrupt:
        print("\nDashboard stopped.")

