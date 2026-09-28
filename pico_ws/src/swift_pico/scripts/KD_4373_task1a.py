#!/usr/bin/env python3
import argparse
import os
import sys
import cv2
import numpy as np

REQUIRED_MARKER_IDS = (80, 85, 90, 95)
CANVAS = 900
CELL = 75
MIN_AREA = 400
MAX_AREA = 8000


def detect_markers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=parameters
        )

    return corners, ids


def select_corner_markers(corners, ids):
    required = set(REQUIRED_MARKER_IDS)

    if ids is None or len(ids) == 0:
        print(f"Error: No ArUco markers detected (required: {sorted(required)})",
              file=sys.stderr)
        return None, None

    ids_flat = [int(m) for m in np.asarray(ids).flatten()]

    if len(ids_flat) != len(required):
        print(f"Error: Detected {len(ids_flat)} markers {sorted(ids_flat)}; "
              f"need exactly {len(required)} ({sorted(required)})", file=sys.stderr)
        return None, None

    if set(ids_flat) != required:
        print(f"Error: Detected IDs {sorted(ids_flat)}; required {sorted(required)}",
              file=sys.stderr)
        return None, None

    return list(corners), np.array(ids_flat)


def order_corner_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def get_arena_corners(marker_corners):
    centers = [np.mean(c[0], axis=0) for c in marker_corners]
    centroid = np.mean(centers, axis=0)

    inner_corners = []
    for corners in marker_corners:
        pts = corners[0]
        distances = [np.linalg.norm(pt - centroid) for pt in pts]
        inner_corners.append(pts[int(np.argmin(distances))])

    inner_corners = np.array(inner_corners, dtype=np.float32)
    return order_corner_points(inner_corners)


def perspective_transform(image, src_pts):
    dst_pts = np.array([
        [0, 0],
        [CANVAS, 0],
        [CANVAS, CANVAS],
        [0, CANVAS]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, matrix, (CANVAS, CANVAS))


def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_survivors(warped):
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 120, 100]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 120, 100]), np.array([180, 255, 255])),
    )

    mask_yellow = cv2.inRange(hsv, np.array([22, 150, 150]), np.array([35, 255, 255]))

    critical_centers = extract_centers(clean_mask(mask_red))
    stable_centers = extract_centers(clean_mask(mask_yellow))

    return critical_centers, stable_centers


def extract_centers(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
    return centers


def map_to_grid(centers):
    labels = []
    for cx, cy in centers:
        col = int(round((cx - CELL) / CELL))
        row = int(round((cy - CELL) / CELL))
        col = max(0, min(10, col))
        row = max(0, min(10, row))
        labels.append(chr(65 + col) + str(row + 1))
    return sorted(set(labels))


def results_path(image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(os.path.dirname(image_path) or '.', f"{stem}_results.txt")


def write_results(image_path, detected_ids, critical_labels, stable_labels):
    output_path = results_path(image_path)
    with open(output_path, 'w') as f:
        f.write(f"Detected marker IDs: {detected_ids}\n\n")
        f.write(f"Critical Survivors: {', '.join(critical_labels)}\n")
        f.write(f"Stable Survivors: {', '.join(stable_labels)}\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Detect survivors in arena image')
    parser.add_argument('--image', required=True, help='Path to input image')
    args = parser.parse_args()

    out = results_path(args.image)
    if os.path.exists(out):
        os.remove(out)

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Cannot load image: {args.image}", file=sys.stderr)
        sys.exit(1)

    corners, ids = detect_markers(image)
    selected_corners, selected_ids = select_corner_markers(corners, ids)

    if selected_corners is None:
        sys.exit(1)

    detected_ids = sorted(int(mid) for mid in selected_ids)

    arena_corners = get_arena_corners(selected_corners)
    warped = perspective_transform(image, arena_corners)

    critical_centers, stable_centers = detect_survivors(warped)

    critical_labels = map_to_grid(critical_centers)
    stable_labels = map_to_grid(stable_centers)

    output_path = write_results(args.image, detected_ids, critical_labels, stable_labels)
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
