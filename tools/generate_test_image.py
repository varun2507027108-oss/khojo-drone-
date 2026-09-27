#!/usr/bin/env python3
"""Generate synthetic Task 1A arena images with real ArUco markers.

The geometry is the one KD_*_task1a.py assumes, so the label you draw is the label
the pipeline reports: a 900 px arena ruled every 75 px, survivors painted on grid
intersections (A1 at 75, 75 ... K11 at 825, 825), and the corner markers outside
the arena with the marker corner that touches it landing exactly on an arena
corner.  Markers keep a margin from the canvas border (ArUco discards candidates
closer than `minDistanceToBorder`).

Usage:
    python tools/generate_test_image.py --out images/test_flat.jpg --critical C4,F2 --stable G8,D6 --no-perspective
    python tools/generate_test_image.py --out images/test_tilted.jpg --critical C4,F2 --stable G8,D6
"""
import argparse
import os

import cv2
import numpy as np

ARENA_SIZE = 900
CELL_SIZE = 75            # must match map_to_grid() in KD_*_task1a.py
GRID_ORIGIN = 75          # labelled intersections start one cell inside the arena
GRID_LINES = 12           # 12 x 12 cells -> 11 x 11 labels, A..K and 1..11
MAX_COL = 10              # 'K'
MAX_ROW = 10              # '11'

MARKER_IDS = (80, 85, 90, 95)   # TL, TR, BR, BL
MARKER_SIZE = 100
MARKER_PAD = 20                 # gap between marker and canvas border
MARGIN = MARKER_SIZE + MARKER_PAD

FLOOR_COLOR = (180, 100, 40)    # blue mud -> saturation/hue far from red and yellow
GRID_COLOR = (255, 255, 255)    # white -> saturation 0, invisible to the colour masks
BACKGROUND = (50, 50, 50)
CRITICAL_COLOR = (0, 0, 255)    # red triangle
STABLE_COLOR = (0, 255, 255)    # yellow circle
OBSTACLE_COLOR = (0, 0, 0)      # black square
FOLIAGE_COLOR = (0, 200, 0)     # green pentagon (hue 60, never matched)

SURVIVOR_SIZE = 25
OBSTACLE_SIZE = 60


def arena_origin():
    return MARGIN, MARGIN


def label_to_pixel(label, origin=None):
    """Pixel of the labelled grid intersection, e.g. 'C4' -> (225, 300) + origin."""
    text = label.strip().upper()
    if len(text) < 2 or not text[0].isalpha() or not text[1:].isdigit():
        raise ValueError(f"bad label {label!r}, expected something like 'C4'")
    col = ord(text[0]) - ord('A')
    row = int(text[1:]) - 1
    if not 0 <= col <= MAX_COL or not 0 <= row <= MAX_ROW:
        raise ValueError(
            f"label {label!r} is outside the labelled grid "
            f"(A..{chr(ord('A') + MAX_COL)} and 1..{MAX_ROW + 1})")
    ox, oy = arena_origin() if origin is None else origin
    return (ox + GRID_ORIGIN + col * CELL_SIZE, oy + GRID_ORIGIN + row * CELL_SIZE)


def create_aruco_marker(dictionary, marker_id, size):
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    if marker.ndim == 2:
        marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return marker


def create_arena_base(size=ARENA_SIZE, margin=MARGIN):
    canvas = np.full((size + 2 * margin, size + 2 * margin, 3), BACKGROUND, dtype=np.uint8)
    cv2.rectangle(canvas, (margin, margin), (margin + size, margin + size),
                  FLOOR_COLOR, thickness=cv2.FILLED)
    return canvas


def draw_grid(arena, size=ARENA_SIZE, margin=MARGIN, cells=GRID_LINES,
              color=GRID_COLOR, thickness=2):
    for i in range(1, cells):
        pos = margin + i * (size // cells)
        cv2.line(arena, (pos, margin), (pos, margin + size), color, thickness)
        cv2.line(arena, (margin, pos), (margin + size, pos), color, thickness)
    return arena


def place_corner_markers(arena, dictionary, marker_size=MARKER_SIZE, margin=MARGIN,
                         size=ARENA_SIZE):
    """Paste 80/85/90/95 just outside the arena corners (inner corner on the corner)."""
    positions = {
        80: (margin - marker_size, margin - marker_size),   # TL, inner corner bottom-right
        85: (margin + size, margin - marker_size),          # TR, inner corner bottom-left
        90: (margin + size, margin + size),                 # BR, inner corner top-left
        95: (margin - marker_size, margin + size),          # BL, inner corner top-right
    }
    for marker_id, (x, y) in positions.items():
        arena[y:y + marker_size, x:x + marker_size] = create_aruco_marker(
            dictionary, marker_id, marker_size)
    return arena


def place_survivors(arena, critical_labels, stable_labels, size=SURVIVOR_SIZE,
                    origin=None):
    """Red triangles = critical, yellow circles = stable, on labelled intersections."""
    for label in critical_labels:
        cx, cy = label_to_pixel(label, origin)
        pts = np.array([[cx, cy - size], [cx - size, cy + size], [cx + size, cy + size]],
                       dtype=np.int32)
        cv2.fillPoly(arena, [pts], CRITICAL_COLOR)

    for label in stable_labels:
        cx, cy = label_to_pixel(label, origin)
        cv2.circle(arena, (cx, cy), size, STABLE_COLOR, thickness=cv2.FILLED)

    return arena


def place_obstacles(arena, count=3, size=OBSTACLE_SIZE, seed=42, origin=None):
    """Black squares + green foliage inside the arena; drawn before the survivors."""
    rng = np.random.RandomState(seed)
    ox, oy = arena_origin() if origin is None else origin
    low_x, low_y = ox + size, oy + size
    high_x, high_y = ox + ARENA_SIZE - size, oy + ARENA_SIZE - size

    for _ in range(count):
        x = int(rng.randint(low_x, high_x))
        y = int(rng.randint(low_y, high_y))
        cv2.rectangle(arena, (x, y), (x + size, y + size), OBSTACLE_COLOR, cv2.FILLED)

    for _ in range(count):
        x = int(rng.randint(low_x, high_x))
        y = int(rng.randint(low_y, high_y))
        pts = np.array([[x, y], [x + 50, y], [x + 25, y + 40],
                        [x - 25, y + 40], [x - 50, y]], dtype=np.int32)
        cv2.fillPoly(arena, [pts], FOLIAGE_COLOR)

    return arena


def apply_perspective_distortion(image, strength=0.15, seed=42):
    """Tilt the whole canvas (markers included) onto a slightly larger canvas."""
    height, width = image.shape[:2]
    dx = int(width * strength)
    dy = int(height * strength)
    rng = np.random.RandomState(seed)

    def jitter(amount):
        return int(rng.randint(-amount // 2, amount // 2 + 1))

    src_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    dst_pts = np.float32([
        [dx + jitter(dx // 2), dy * 2],
        [width + dx, dy + jitter(dy // 2)],
        [width + dx // 2, height + dy],
        [dx * 2, height + dy // 2],
    ])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, matrix, (width + dx * 3, height + dy * 3),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=BACKGROUND)


def generate_test_image(output_path, critical_labels, stable_labels,
                        perspective=True, seed=42, obstacles=True):
    """Compose the arena, optionally tilt it, write it, and report the expected answer."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    arena = create_arena_base()
    arena = draw_grid(arena)
    arena = place_corner_markers(arena, dictionary)
    if obstacles:
        arena = place_obstacles(arena, seed=seed)
    arena = place_survivors(arena, critical_labels, stable_labels)
    if perspective:
        arena = apply_perspective_distortion(arena, seed=seed)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if not cv2.imwrite(output_path, arena):
        raise SystemExit(f"Error: Cannot write image: {output_path}")

    expected = {
        'Detected marker IDs': list(MARKER_IDS),
        'Critical Survivors': sorted(critical_labels),
        'Stable Survivors': sorted(stable_labels),
    }

    print(f"Generated: {output_path} ({arena.shape[1]}x{arena.shape[0]}, "
          f"{'tilted' if perspective else 'flat'})")
    print("Expected output:")
    print(f"  Detected marker IDs: {expected['Detected marker IDs']}")
    print(f"  Critical Survivors: {', '.join(expected['Critical Survivors'])}")
    print(f"  Stable Survivors: {', '.join(expected['Stable Survivors'])}")

    return output_path


def split_labels(text):
    return [part.strip().upper() for part in text.split(',') if part.strip()]


def main():
    parser = argparse.ArgumentParser(description='Generate a test arena image with real ArUco markers')
    parser.add_argument('--out', required=True, help='Output image path')
    parser.add_argument('--critical', default='C4,F2', help='Critical survivor labels (comma-separated)')
    parser.add_argument('--stable', default='G8,D6', help='Stable survivor labels (comma-separated)')
    parser.add_argument('--no-perspective', action='store_true', help='Generate flat (top-down) image')
    parser.add_argument('--no-obstacles', action='store_true', help='Leave out black squares and foliage')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for obstacle placement')
    args = parser.parse_args()

    critical = split_labels(args.critical)
    stable = split_labels(args.stable)
    try:
        for label in critical + stable:
            label_to_pixel(label)
    except ValueError as error:
        raise SystemExit(f"Error: {error}")

    generate_test_image(args.out, critical, stable,
                        perspective=not args.no_perspective,
                        seed=args.seed,
                        obstacles=not args.no_obstacles)


if __name__ == '__main__':
    main()
