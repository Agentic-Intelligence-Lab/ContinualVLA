#!/usr/bin/env python3
"""Monitor GPU utilization and auto-resume training when idle for 5 minutes."""

import subprocess
import time
import os
import sys

CHECK_INTERVAL = 30  # seconds between checks
IDLE_THRESHOLD = 300  # 5 minutes in seconds
RESUME_SCRIPT = os.path.join(os.path.dirname(__file__), "train_cl_resume.sh")


def get_gpu_utilizations():
    """Return list of GPU utilization percentages, or None on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None
        return [int(line.strip()) for line in result.stdout.strip().split("\n") if line.strip()]
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        return None


def main():
    idle_since = None  # timestamp when all GPUs first hit 0
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"[gpu_monitor] Watching GPUs every {CHECK_INTERVAL}s, "
          f"will run {RESUME_SCRIPT} after {IDLE_THRESHOLD}s of idle")

    while True:
        utils = get_gpu_utilizations()

        if utils is None:
            print(f"[gpu_monitor] nvidia-smi query failed at {time.strftime('%H:%M:%S')}, retrying...")
            time.sleep(CHECK_INTERVAL)
            continue

        max_util = max(utils) if utils else 0
        now = time.time()

        if max_util <= 1:
            if idle_since is None:
                idle_since = now
            elapsed = now - idle_since
            print(f"[gpu_monitor] {time.strftime('%H:%M:%S')} GPUs idle for {int(elapsed)}s "
                  f"(threshold: {IDLE_THRESHOLD}s)")
            if elapsed >= IDLE_THRESHOLD:
                print(f"[gpu_monitor] GPU idle for {int(elapsed)}s >= {IDLE_THRESHOLD}s, "
                      f"running {RESUME_SCRIPT}...")
                try:
                    subprocess.run(
                        ["bash", RESUME_SCRIPT],
                        cwd=os.path.dirname(script_dir),
                        check=True,
                        timeout=600,
                    )
                    print(f"[gpu_monitor] Script completed successfully at {time.strftime('%H:%M:%S')}")
                except subprocess.CalledProcessError as e:
                    print(f"[gpu_monitor] Script failed with exit code {e.returncode}")
                except subprocess.TimeoutExpired:
                    print(f"[gpu_monitor] Script timed out after 600s")
                # Reset after running so we don't re-trigger immediately
                idle_since = None
        else:
            if idle_since is not None:
                print(f"[gpu_monitor] {time.strftime('%H:%M:%S')} GPU active again (max={max_util}%), resetting idle timer")
            idle_since = None

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
