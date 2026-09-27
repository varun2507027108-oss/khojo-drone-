#!/usr/bin/env python3
import argparse
import os
import sys
import cv2
import numpy as np


def detect_markers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


REQUIRED_MARKER_IDS = (80, 85, 90, 95)


def select_corner_markers(corners, ids, required_ids=None):
    required = set(REQUIRED_MARKER_IDS if required_ids is None else required_ids)

    if ids is None:
        print(f"Error: No ArUco markers detected (required: {sorted(required)})")
        return None, None

    ids_flat = [int(marker_id) for marker_id in np.asarray(ids).flatten()]
    detected_set = set(ids_flat)

    missing = required - detected_set
    if missing:
        print(f"Error: Missing required marker IDs: {sorted(missing)}")
        return None, None

    selected_corners = []
    selected_ids = []
    for marker_corners, marker_id in zip(corners, ids_flat):
        if marker_id in required:
            selected_corners.append(marker_corners)
            selected_ids.append(marker_id)

    return selected_corners, np.array(selected_ids)


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
    centers = []
    for corners in marker_corners:
        center = np.mean(corners[0], axis=0)
        centers.append(center)
    centroid = np.mean(centers, axis=0)

    inner_corners = []
    for corners in marker_corners:
        pts = corners[0]
        distances = [np.linalg.norm(pt - centroid) for pt in pts]
        inner_idx = np.argmin(distances)
        inner_corners.append(pts[inner_idx])

    inner_corners = np.array(inner_corners, dtype=np.float32)
    return order_corner_points(inner_corners)


def perspective_transform(image, src_pts):
    dst_pts = np.array([
        [0, 0],
        [900, 0],
        [900, 900],
        [0, 900]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, matrix, (900, 900))
    return warped


def detect_survivors(warped):
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

    critical_centers = extract_centers(mask_red)
    stable_centers = extract_centers(mask_yellow)

    return critical_centers, stable_centers


def extract_centers(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
    return centers


def map_to_grid(centers):
    labels = []
    cell_size = 75
    for cx, cy in centers:
        col = int(round((cx - 75) / cell_size))
        row = int(round((cy - 75) / cell_size))
        col = max(0, min(10, col))
        row = max(0, min(10, row))
        label = chr(65 + col) + str(row + 1)
        labels.append(label)
    return labels


def write_results(image_path, detected_ids, critical_labels, stable_labels):
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.dirname(image_path) or '.'
    output_path = os.path.join(output_dir, f"{image_stem}_results.txt")

    with open(output_path, 'w') as f:
        f.write(f"Detected marker IDs: {detected_ids}\n\n")
        f.write(f"Critical Survivors: {', '.join(critical_labels)}\n")
        f.write(f"Stable Survivors: {', '.join(stable_labels)}\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Detect survivors in arena image')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--required-ids',
                        default=','.join(str(marker_id) for marker_id in REQUIRED_MARKER_IDS),
                        help='Comma-separated marker IDs marking the four arena corners')
    args = parser.parse_args()

    try:
        required_ids = {int(part) for part in args.required_ids.split(',') if part.strip()}
    except ValueError:
        print(f"Error: --required-ids must be comma-separated integers: {args.required_ids}")
        sys.exit(1)
    if not required_ids:
        print("Error: --required-ids must contain at least one marker ID")
        sys.exit(1)

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Cannot load image: {args.image}")
        sys.exit(1)

    corners, ids = detect_markers(image)
    selected_corners, selected_ids = select_corner_markers(corners, ids, required_ids)

    if selected_corners is None:
        print("Error: Need all four required ArUco markers: "
              + ', '.join(str(marker_id) for marker_id in sorted(required_ids)))
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
