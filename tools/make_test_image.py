#!/usr/bin/env python3
"""Generate a synthetic Task 1A arena image so the detector can be smoke-tested.

Four DICT_4X4_250 markers (80/85/90/95 by default) sit just outside the arena
corners with their inner corner on the corner, and one red (critical) and one
yellow (stable) blob sit inside.

Usage:
    python tools/make_test_image.py --out images/image_1.jpg
"""
import argparse
import os
import cv2
import numpy as np

# Source-image geometry (all in pixels).
CANVAS = (960, 800)                 # width, height
MARKER_PX = 60                      # rendered marker side length
ARENA = (200, 150, 800, 650)        # x1, y1, x2, y2 = the four inner marker corners
ARENA_COLOR = (210, 210, 210)       # light grey mat -> saturation 0, matches no colour mask
ARENA_BORDER = (60, 60, 60)         # dark boundary line drawn inside the arena edge
ARENA_BORDER_INSET = 8              # keeps the line clear of the corner markers
ARENA_BORDER_THICKNESS = 4
BACKGROUND = (255, 255, 255)        # white background = quiet zone for the markers
MARKER_IDS = (80, 85, 90, 95)       # default TL, TR, BR, BL corner marker IDs
CRITICAL_CELLS = ("C4",)            # red blobs
STABLE_CELLS = ("G8",)              # yellow blobs
BLOB_RADIUS = 15                    # radius in source pixels

WARP_SIZE = 900                     # the script warps the arena to 900 x 900
CELL_SIZE = 75                      # grid pitch used by map_to_grid()
GRID_ORIGIN = 75                    # pixel centre of cell A1 in the warped image


def cell_to_source(label, arena=ARENA):
    """Convert a grid label such as 'C4' into source-image pixel coordinates."""
    col = ord(label[0].upper()) - ord('A')
    row = int(label[1:]) - 1
    x_warped = GRID_ORIGIN + CELL_SIZE * col
    y_warped = GRID_ORIGIN + CELL_SIZE * row
    x1, y1, x2, y2 = arena
    sx = x1 + x_warped * (x2 - x1) / WARP_SIZE
    sy = y1 + y_warped * (y2 - y1) / WARP_SIZE
    return int(round(sx)), int(round(sy))


def paste_marker(image, marker_id, inner_corner, direction):
    """Paste a marker so that its `direction` corner lands on `inner_corner`."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_PX)
    marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    x, y = inner_corner
    dx, dy = direction
    if dx > 0:
        left = x
    else:
        left = x - MARKER_PX
    if dy > 0:
        top = y
    else:
        top = y - MARKER_PX

    image[top:top + MARKER_PX, left:left + MARKER_PX] = marker
    return image


def build_image():
    width, height = CANVAS
    image = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)

    x1, y1, x2, y2 = ARENA
    cv2.rectangle(image, (x1, y1), (x2, y2), ARENA_COLOR, thickness=cv2.FILLED)
    cv2.rectangle(image,
                  (x1 + ARENA_BORDER_INSET, y1 + ARENA_BORDER_INSET),
                  (x2 - ARENA_BORDER_INSET, y2 - ARENA_BORDER_INSET),
                  ARENA_BORDER, thickness=ARENA_BORDER_THICKNESS)

    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    directions = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for marker_id, corner, direction in zip(MARKER_IDS, corners, directions):
        paste_marker(image, marker_id, corner, direction)

    for label in CRITICAL_CELLS:
        cv2.circle(image, cell_to_source(label), BLOB_RADIUS, (0, 0, 255), thickness=cv2.FILLED)
    for label in STABLE_CELLS:
        cv2.circle(image, cell_to_source(label), BLOB_RADIUS, (0, 255, 255), thickness=cv2.FILLED)

    return image


def erase_marker(image, index, pad=8):
    """White out marker gen.MARKER_IDS[index] (plus its quiet zone) to simulate a lost marker."""
    x1, y1, x2, y2 = ARENA
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    directions = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    x, y = corners[index]
    dx, dy = directions[index]
    left = x if dx > 0 else x - MARKER_PX
    top = y if dy > 0 else y - MARKER_PX
    height, width = image.shape[:2]
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(width, left + MARKER_PX + 2 * pad)
    bottom = min(height, top + MARKER_PX + 2 * pad)
    image[top:bottom, left:right] = BACKGROUND
    return image


def erase_marker_id(image, marker_id, pad=8):
    """White out the corner marker with the given ID."""
    if marker_id not in MARKER_IDS:
        raise ValueError(f"Unknown corner marker ID {marker_id}; expected one of {list(MARKER_IDS)}")
    return erase_marker(image, MARKER_IDS.index(marker_id), pad)


def parse_marker_ids(text):
    try:
        marker_ids = [int(part.strip()) for part in text.split(',') if part.strip()]
    except ValueError as error:
        raise ValueError(f"bad marker ID list {text!r}: {error}")
    if len(marker_ids) != 4:
        raise ValueError(f"need exactly 4 marker IDs, got {marker_ids}")
    if len(set(marker_ids)) != 4:
        raise ValueError(f"marker IDs must be distinct, got {marker_ids}")
    return tuple(marker_ids)


def main():
    parser = argparse.ArgumentParser(description='Create a synthetic Task 1A arena image')
    parser.add_argument('--out', default=os.path.join('images', 'image_1.jpg'),
                        help='Output image path (default: images/image_1.jpg)')
    parser.add_argument('--erase-id', type=int, default=None,
                        help='Erase one corner marker ID to test the missing-marker abort path')
    parser.add_argument('--marker-ids',
                        default=','.join(str(marker_id) for marker_id in MARKER_IDS),
                        help='Exactly four comma-separated ArUco marker IDs in TL,TR,BR,BL order')
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    global MARKER_IDS
    try:
        MARKER_IDS = parse_marker_ids(args.marker_ids)
    except ValueError as error:
        raise SystemExit(f"Error: {error}")

    image = build_image()
    if args.erase_id is not None:
        image = erase_marker_id(image, args.erase_id)

    if not cv2.imwrite(args.out, image):
        raise SystemExit(f"Error: Cannot write image: {args.out}")

    print(f"Wrote {args.out} ({image.shape[1]}x{image.shape[0]})")
    if args.erase_id is not None:
        print(f"Marker {args.erase_id} erased -> the detector must abort")
        return

    print(f"Expected marker IDs: {list(MARKER_IDS)}")
    print(f"Expected Critical Survivors: {', '.join(CRITICAL_CELLS)}")
    print(f"Expected Stable Survivors: {', '.join(STABLE_CELLS)}")


if __name__ == '__main__':
    main()
