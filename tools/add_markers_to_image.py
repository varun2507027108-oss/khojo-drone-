#!/usr/bin/env python3
"""Add four Task 1A arena markers to a picture that has none.

Markers go outside the arena on a dark plate inside an added canvas margin, each
marker image corner landing exactly on an arena corner (never flush to the canvas
border: ArUco discards candidates closer than `minDistanceToBorder`).  The arena
defaults to a 12 x 12 cell window of the picture's own grid -- 12 cells keeps the
75 px pitch that map_to_grid() assumes -- offset so the survivor blobs land as
close as possible to labelled intersections.  The output image is decoded again
before exit, and the exit status is 1 if the markers do not define the arena.

Usage:
    python tools/add_markers_to_image.py --image images/my_arena.png
    python tools/add_markers_to_image.py --image images/my_arena.png --arena 160,156,1176,1117
    python tools/add_markers_to_image.py --image images/my_arena.png --preview
"""
import argparse
import os
import sys

import cv2
import numpy as np

# Marker geometry, same recipe as tools/generate_test_image.py.
MARKER_IDS = (80, 85, 90, 95)      # default TL, TR, BR, BL
MARKER_SIZE = 120                  # divisible by 6 so the marker modules stay pixel-exact
PLATE_PAD = 20                     # dark plate around each marker
BORDER_GAP = 12                    # backdrop kept between plate and canvas border
BACKDROP = (50, 50, 50)            # dark grey
DICTIONARY_ID = cv2.aruco.DICT_4X4_250
MODULES_PER_SIDE = 6               # 4x4 data + one black border module per side

# Arena geometry, kept in sync with KD_*_task1a.py.
ARENA_CELLS = 12                   # cells across the marked arena -> 900 / 12 = 75 px pitch
WARP_SIZE = 900
CELL_SIZE = 75
GRID_ORIGIN = 75
MAX_INDEX = 10                     # last label map_to_grid() can emit: 'K', '11'

# A painted grid line is a bright, unsaturated run across the picture.
GRID_MIN_FRACTION = 0.35           # a grid line must cover >= 35 % of the image
GRID_BRIGHT_VALUE = 200
GRID_MAX_SATURATION = 90
OFFSET_STEPS = 40                  # sub-cell arena offsets tried per axis (1/40 cell = ~2 px)

# Survivor masks, copied from detect_survivors() in KD_*_task1a.py.
CRITICAL_HSV = ((0, 120, 70, 10, 255, 255), (170, 120, 70, 180, 255, 255))
STABLE_HSV = ((20, 100, 100, 35, 255, 255),)
MIN_BLOB_AREA = 100


def find_survivors(image):
    """Return {'critical': [(x, y), ...], 'stable': [(x, y), ...]} in source pixels."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    survivors = {}
    for name, ranges in (('critical', CRITICAL_HSV), ('stable', STABLE_HSV)):
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo_h, lo_s, lo_v, hi_h, hi_s, hi_v in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv,
                                                   np.array([lo_h, lo_s, lo_v]),
                                                   np.array([hi_h, hi_s, hi_v])))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        centres = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < MIN_BLOB_AREA:
                continue
            moments = cv2.moments(contour)
            if moments['m00']:
                centres.append((moments['m10'] / moments['m00'],
                                moments['m01'] / moments['m00']))
        survivors[name] = centres
    return survivors


def _bright_line_peaks(profile, min_fraction):
    """Centre pixel of every run of `profile` above `min_fraction` (one grid line)."""
    peaks = []
    index = 0
    while index < len(profile):
        if profile[index] <= min_fraction:
            index += 1
            continue
        end = index
        while end < len(profile) and profile[end] > min_fraction:
            end += 1
        peaks.append(index + int(np.argmax(profile[index:end])))
        index = end
    return peaks


def find_grid_lines(image):
    """Locate the picture's own grid lines -> (column pixels, row pixels)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bright = ((hsv[:, :, 2] > GRID_BRIGHT_VALUE) &
              (hsv[:, :, 1] < GRID_MAX_SATURATION)).astype(np.float32)
    columns = _bright_line_peaks(bright.mean(axis=0), GRID_MIN_FRACTION)
    rows = _bright_line_peaks(bright.mean(axis=1), GRID_MIN_FRACTION)
    return columns, rows


def label_and_margin(x, y, arena):
    """Label map_to_grid() will report for a point, plus how safe that label is.

    The second value is the distance to the position whose label the point gets,
    in cells: 0 = dead centre, 0.5 = on the boundary where the label flips.
    """
    x1, y1, x2, y2 = arena
    warped_x = (x - x1) * WARP_SIZE / (x2 - x1)
    warped_y = (y - y1) * WARP_SIZE / (y2 - y1)
    col = int(round((warped_x - GRID_ORIGIN) / CELL_SIZE))
    row = int(round((warped_y - GRID_ORIGIN) / CELL_SIZE))
    label = (chr(ord('A') + max(0, min(MAX_INDEX, col))) +
             str(max(0, min(MAX_INDEX, row)) + 1))
    distance = max(abs((warped_x - GRID_ORIGIN) / CELL_SIZE - col),
                   abs((warped_y - GRID_ORIGIN) / CELL_SIZE - row))
    return label, distance


def choose_arena(columns, rows, survivors, bounds):
    """Best arena rectangle -> (arena, blobs inside, worst distance to a label).

    Candidates are ranked by most blobs inside, then smallest worst-case distance
    to a labelled intersection, then closest to the middle of the picture.
    """
    width, height = bounds
    points = [point for group in survivors.values() for point in group]
    best = None
    for i in range(len(columns) - ARENA_CELLS):
        span_x = columns[i + ARENA_CELLS] - columns[i]
        for j in range(len(rows) - ARENA_CELLS):
            span_y = rows[j + ARENA_CELLS] - rows[j]
            for step_x in range(OFFSET_STEPS):
                x1 = columns[i] + span_x * step_x / OFFSET_STEPS
                x2 = x1 + span_x
                if x1 < 0 or x2 > width:
                    continue
                for step_y in range(OFFSET_STEPS):
                    y1 = rows[j] + span_y * step_y / OFFSET_STEPS
                    y2 = y1 + span_y
                    if y1 < 0 or y2 > height:
                        continue
                    arena = (x1, y1, x2, y2)
                    inside = [p for p in points if x1 <= p[0] <= x2 and y1 <= p[1] <= y2]
                    worst = max((label_and_margin(p[0], p[1], arena)[1] for p in inside),
                                default=1.0)
                    offset = abs((x1 + x2) / 2 - width / 2) + abs((y1 + y2) / 2 - height / 2)
                    score = (-len(inside), worst, offset)
                    if best is None or score < best[0]:
                        best = (score, arena, len(inside), worst)
    if best is None:
        return None
    return best[1], best[2], best[3]


def paste(canvas, patch, box):
    """Copy `patch` into `canvas` at (x1, y1, x2, y2), clipped to the canvas."""
    height, width = canvas.shape[:2]
    x1, y1, x2, y2 = box
    left, top = max(0, x1), max(0, y1)
    right, bottom = min(width, x2), min(height, y2)
    if left >= right or top >= bottom:
        return canvas
    canvas[top:bottom, left:right] = patch[top - y1:bottom - y1, left - x1:right - x1]
    return canvas


def paste_markers(canvas, arena, dictionary, marker_ids, marker_size):
    """Paste four markers outside the arena so they define the arena corners."""
    x1, y1, x2, y2 = (int(round(value)) for value in arena)
    size = marker_size        # the marker image is its black square
    tops = {
        marker_ids[0]: (x1 - size, y1 - size),
        marker_ids[1]: (x2, y1 - size),
        marker_ids[2]: (x2, y2),
        marker_ids[3]: (x1 - size, y2),
    }
    plates = {
        marker_ids[0]: (x1 - size - PLATE_PAD, y1 - size - PLATE_PAD, x1, y1),
        marker_ids[1]: (x2, y1 - size - PLATE_PAD, x2 + size + PLATE_PAD, y1),
        marker_ids[2]: (x2, y2, x2 + size + PLATE_PAD, y2 + size + PLATE_PAD),
        marker_ids[3]: (x1 - size - PLATE_PAD, y2, x1, y2 + size + PLATE_PAD),
    }

    for marker_id in marker_ids:
        px1, py1, px2, py2 = plates[marker_id]
        paste(canvas, np.full((py2 - py1, px2 - px1, 3), BACKDROP, dtype=np.uint8),
              plates[marker_id])

    for marker_id in marker_ids:
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
        if marker.ndim == 2:
            marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        left, top = tops[marker_id]
        paste(canvas, marker, (left, top, left + marker_size, top + marker_size))
    return canvas


def detect_markers(image):
    """Same detector settings as KD_*_task1a.py detect_markers()."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(DICTIONARY_ID), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def marker_targets(arena, marker_ids):
    """Arena corner each marker's black square must touch (TL, TR, BR, BL)."""
    x1, y1, x2, y2 = arena
    return {
        marker_ids[0]: (x1, y1),
        marker_ids[1]: (x2, y1),
        marker_ids[2]: (x2, y2),
        marker_ids[3]: (x1, y2),
    }


def verify_markers(canvas, arena, marker_ids, tolerance=4.0):
    """Decode the pasted markers and check the arena rectangle they define."""
    corners, ids = detect_markers(canvas)
    found = [] if ids is None else sorted(int(i) for i in np.asarray(ids).flatten())
    print(f"  decoded marker IDs: {found}")
    if found != sorted(marker_ids):
        print(f"  FAIL  expected exactly {sorted(marker_ids)}")
        return False

    targets = marker_targets(arena, marker_ids)
    ok = True
    for marker_corners, marker_id in zip(corners, np.asarray(ids).flatten()):
        target = np.array(targets[int(marker_id)], dtype=np.float32)
        deviation = min(float(np.linalg.norm(point - target))
                        for point in marker_corners[0])
        if deviation > tolerance:
            ok = False
        print(f"  marker {int(marker_id)}: arena corner "
              f"({target[0]:.0f}, {target[1]:.0f}) off by {deviation:.1f} px -> "
              f"{'ok' if deviation <= tolerance else 'FAIL'}")
    return ok


def write_image(image, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if path.lower().endswith(('.jpg', '.jpeg')) else []
    if not cv2.imwrite(path, image, params):
        raise SystemExit(f"Error: Cannot write image: {path}")
    return path


def write_preview(canvas, arena, arena_shifted, survivors, shift, path, marker_ids):
    """Draw the arena, the survivor blobs and the labels they are expected to get."""
    preview = canvas.copy()
    x1, y1, x2, y2 = arena_shifted
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)

    for name, colour in (('critical', (0, 0, 255)), ('stable', (0, 255, 255))):
        for x, y in survivors[name]:
            centre = (int(round(x + shift[0])), int(round(y + shift[1])))
            label = label_and_margin(x, y, arena)[0]
            cv2.circle(preview, centre, 40, colour, 3)
            origin = (centre[0] + 30, centre[1] - 30)
            cv2.putText(preview, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5)
            cv2.putText(preview, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 2)

    for marker_id, (tx, ty) in marker_targets(arena_shifted, marker_ids).items():
        cv2.circle(preview, (int(tx), int(ty)), 12, (255, 255, 255), 2)

    return write_image(preview, path)


def parse_arena(text):
    parts = [part.strip() for part in text.split(',')]
    if len(parts) != 4:
        raise ValueError(f"expected four comma-separated pixels (x1,y1,x2,y2), got {text!r}")
    values = tuple(int(part) for part in parts)
    if not (values[0] < values[2] and values[1] < values[3]):
        raise ValueError(f"expected x1 < x2 and y1 < y2, got {text!r}")
    return values


def parse_marker_ids(text):
    parts = [part.strip() for part in text.split(',') if part.strip()]
    try:
        marker_ids = tuple(int(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"bad marker ID list {text!r}: {error}")
    if len(marker_ids) != 4:
        raise ValueError(f"need exactly four marker IDs, got {list(marker_ids)}")
    if len(set(marker_ids)) != 4:
        raise ValueError(f"marker IDs must be distinct, got {list(marker_ids)}")
    return marker_ids


def default_output_path(image_path):
    stem, extension = os.path.splitext(image_path)
    return f"{stem}_with_markers{extension or '.png'}"


def main():
    parser = argparse.ArgumentParser(
        description='Add four Task 1A corner markers to an existing picture')
    parser.add_argument('--image', required=True,
                        help='Input picture (top-down arena photo with a visible grid)')
    parser.add_argument('--out', default=None,
                        help='Output picture (default: <image stem>_with_markers<ext>)')
    parser.add_argument('--arena', default=None,
                        help='Arena rectangle in source pixels x1,y1,x2,y2 '
                             f'(default: {ARENA_CELLS} x {ARENA_CELLS} cells of the picture grid)')
    parser.add_argument('--marker-size', type=int, default=MARKER_SIZE,
                        help=f'Marker PNG side in pixels, a multiple of {MODULES_PER_SIDE} '
                             f'(default: {MARKER_SIZE})')
    parser.add_argument('--preview', action='store_true',
                        help='Also write <out stem>_preview with the arena and expected labels')
    parser.add_argument('--marker-ids',
                        default=','.join(str(marker_id) for marker_id in MARKER_IDS),
                        help='Exactly four comma-separated ArUco marker IDs in TL,TR,BR,BL order')
    args = parser.parse_args()

    if args.marker_size <= 0 or args.marker_size % MODULES_PER_SIDE:
        print(f"Error: --marker-size must be a positive multiple of {MODULES_PER_SIDE}")
        sys.exit(1)
    try:
        marker_ids = parse_marker_ids(args.marker_ids)
    except ValueError as error:
        print(f"Error: --marker-ids {error}")
        sys.exit(1)

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Cannot load image: {args.image}")
        sys.exit(1)

    height, width = image.shape[:2]
    survivors = find_survivors(image)
    print(f"Input: {args.image} ({width}x{height})")
    print("Survivor blobs in the picture: "
          f"critical={len(survivors['critical'])}, stable={len(survivors['stable'])}")

    if args.arena:
        try:
            arena = parse_arena(args.arena)
        except ValueError as error:
            print(f"Error: --arena {error}")
            sys.exit(1)
        source = 'from --arena'
    else:
        columns, rows = find_grid_lines(image)
        print(f"Picture grid detected: {len(columns)} columns, {len(rows)} rows")
        chosen = choose_arena(columns, rows, survivors, (width, height))
        if chosen is None:
            print(f"Error: no {ARENA_CELLS}-cell grid window found in the picture")
            print("       pass --arena x1,y1,x2,y2, or use tools/generate_test_image.py")
            sys.exit(1)
        arena, inside, worst = chosen
        source = (f"auto: {ARENA_CELLS} x {ARENA_CELLS} grid cells with {inside} survivor "
                  f"blob(s) inside, worst blob {worst:.2f} cells from an intersection")

    x1, y1, x2, y2 = arena
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        print(f"Error: arena {arena} is not inside the image ({width}x{height})")
        sys.exit(1)
    arena = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
    x1, y1, x2, y2 = arena
    print(f"Arena: ({x1:.0f}, {y1:.0f})-({x2:.0f}, {y2:.0f}) ({source})")

    margin = args.marker_size + PLATE_PAD + BORDER_GAP
    canvas = np.full((height + 2 * margin, width + 2 * margin, 3), BACKDROP, dtype=np.uint8)
    paste(canvas, image, (margin, margin, margin + width, margin + height))

    arena_shifted = (x1 + margin, y1 + margin, x2 + margin, y2 + margin)
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_ID)
    paste_markers(canvas, arena_shifted, dictionary, marker_ids, args.marker_size)

    out_path = write_image(canvas, args.out or default_output_path(args.image))
    print(f"Wrote {out_path} ({canvas.shape[1]}x{canvas.shape[0]}, "
          f"{margin} px margin per side)")

    print("Marker check (same detector settings as the pipeline):")
    verified = verify_markers(canvas, arena_shifted, marker_ids)

    labels = {'critical': [], 'stable': []}
    print("Survivor labels the pipeline should report:")
    for name in ('critical', 'stable'):
        for x, y in survivors[name]:
            label, distance = label_and_margin(x, y, arena)
            labels[name].append(label)
            warning = '' if distance <= 0.35 else '   (sits near the rounding boundary)'
            print(f"  {name} blob at ({x:.0f}, {y:.0f}) -> {label}, "
                  f"{distance:.2f} cells from the nearest labelled intersection{warning}")

    print("Expected results file:")
    print(f"  Detected marker IDs: {sorted(marker_ids)}")
    print(f"  Critical Survivors: {', '.join(labels['critical'])}")
    print(f"  Stable Survivors: {', '.join(labels['stable'])}")

    if args.preview:
        stem, extension = os.path.splitext(out_path)
        preview_path = write_preview(canvas, arena, arena_shifted, survivors,
                                     (margin, margin), f"{stem}_preview{extension}",
                                     marker_ids)
        print(f"Wrote {preview_path} (arena, blobs, expected labels)")

    if not verified:
        print("Error: the markers on the output do not define the arena cleanly")
        sys.exit(1)


if __name__ == '__main__':
    main()
