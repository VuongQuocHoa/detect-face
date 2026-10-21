"""CLI Runner for real-time camera face recognition.

Allows running directly via:
    python run_camera.py --cam 0
    python run_camera.py --cam 0 --class_name apple
"""

import sys
import argparse
from test_cam import main as test_cam_main


def main() -> None:
    # Pre-parse and normalize arguments so both --cam and --camera work seamlessly
    argv = list(sys.argv[1:])
    normalized_argv = []
    skip_next = False

    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue

        if arg == "--cam":
            normalized_argv.append("--camera")
        elif arg.startswith("--cam="):
            normalized_argv.append("--camera=" + arg.split("=", 1)[1])
        elif arg == "--class_name":
            # Ignore extra object-detection tag if passed by mistake
            skip_next = True
        elif arg.startswith("--class_name="):
            pass
        else:
            normalized_argv.append(arg)

    # Replace sys.argv and call test_cam_main
    sys.argv = [sys.argv[0]] + normalized_argv
    test_cam_main()


if __name__ == "__main__":
    main()
