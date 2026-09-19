#!/usr/bin/env python3
"""Build an approximate fisheye calibration for a real (uncalibrated) photo.

Real internet fisheye photos have no lab calibration file, so we assume the
common equidistant model (r = f*theta) with the optical center at the image
center and fx=fy chosen so the image radius maps to a ~180 deg diagonal FOV.
This mirrors what an installer would do for a first-pass AVM camera before
running proper checkerboard calibration.
"""
import json
import sys
import math
import cv2

def main():
    path = sys.argv[1]
    out = sys.argv[2]
    fov_deg = float(sys.argv[3]) if len(sys.argv) > 3 else 180.0

    img = cv2.imread(path)
    h, w = img.shape[:2]
    r_max = min(w, h) / 2.0
    theta_max = math.radians(fov_deg) / 2.0
    f = r_max / theta_max

    calib = {
        "camera_matrix": [[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]],
        "distortion_coeffs": [0.0, 0.0, 0.0, 0.0],
        "new_camera_matrix": [[f * 0.5, 0, w / 2], [0, f * 0.5, h / 2], [0, 0, 1]],
        "width": w,
        "height": h,
        "fov_degrees": fov_deg,
        "model": "OPENCV_FISHEYE_EQUIDISTANT_ESTIMATED",
    }
    with open(out, "w") as fp:
        json.dump(calib, fp, indent=2)
    print(f"{path}: {w}x{h}, f={f:.1f}, saved {out}")

if __name__ == "__main__":
    main()
