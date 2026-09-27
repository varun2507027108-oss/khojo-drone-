#!/usr/bin/env python3
"""Extra checks for KD_1234_task1a.py: tilted, rotated, crowded and marker-less arenas.

Usage:
    python tools/selftest.py
"""
import contextlib
import importlib.util
import io
import os
import sys
import cv2
import numpy as np

import make_test_image as gen

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.join(os.path.dirname(HERE), 'pico_ws', 'src', 'swift_pico', 'scripts')
SCRIPT_PATH = os.path.join(SCRIPT_DIR, 'KD_1234_task1a.py')


def load_detector():
    spec = importlib.util.spec_from_file_location('kd_task1a', SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_pipeline(module, image):
    corners, ids = module.detect_markers(image)
    selected_corners, selected_ids = module.select_corner_markers(corners, ids)
    if selected_corners is None:
        return None, [], []
    arena_corners = module.get_arena_corners(selected_corners)
    warped = module.perspective_transform(image, arena_corners)
    critical, stable = module.detect_survivors(warped)
    return ([int(i) for i in selected_ids],
            sorted(module.map_to_grid(critical)),
            sorted(module.map_to_grid(stable)))


def check(name, got, expected):
    ok = got == expected
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got}, expected {expected}")
    return ok


def tilt(image):
    """Re-project the flat arena image as if the camera viewed it off-axis."""
    height, width = image.shape[:2]
    src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    dst = np.float32([[70, 45], [width + 40, 8], [width - 15, height + 55], [25, height - 20]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (width + 60, height + 70),
                               borderValue=gen.BACKGROUND)


def add_inner_markers(image):
    """Paste two extra markers inside the arena: they must be ignored by the filter."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    for marker_id, (cx, cy) in ((10, (450, 350)), (11, (600, 470))):
        side = gen.MARKER_PX
        cv2.rectangle(image, (cx - side, cy - side), (cx + side, cy + side),
                      gen.BACKGROUND, thickness=cv2.FILLED)
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
        image[cy - side // 2:cy + side // 2, cx - side // 2:cx + side // 2] = \
            cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return image


def erase_marker(image, index, pad=8):
    """Delegates to the generator so both tools share identical geometry."""
    return gen.erase_marker(image, index, pad)


def main():
    module = load_detector()
    required = sorted(gen.MARKER_IDS)
    results = []

    flat = gen.build_image()
    ids, critical, stable = run_pipeline(module, flat)
    results.append(check('flat image marker IDs', sorted(ids or []), required))
    results.append(check('flat image critical cells', critical, ['C4']))
    results.append(check('flat image stable cells', stable, ['G8']))

    warped_input = tilt(flat)
    ids, critical, stable = run_pipeline(module, warped_input)
    results.append(check('tilted image marker IDs', sorted(ids or []), required))
    results.append(check('tilted image critical cells', critical, ['C4']))
    results.append(check('tilted image stable cells', stable, ['G8']))

    crowded = add_inner_markers(gen.build_image())
    ids, critical, stable = run_pipeline(module, crowded)
    results.append(check('non-required markers ignored', sorted(ids or []), required))
    results.append(check('6-marker image critical cells', critical, ['C4']))
    results.append(check('6-marker image stable cells', stable, ['G8']))

    broken = erase_marker(gen.build_image(), index=3)          # remove ID 95
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ids, critical, stable = run_pipeline(module, broken)
    results.append(check(f'missing ID {gen.MARKER_IDS[3]} aborts',
                         (ids, critical, stable), (None, [], [])))
    results.append(check('abort names only the missing ID',
                         buffer.getvalue().strip(),
                         f"Error: Missing required marker IDs: [{gen.MARKER_IDS[3]}]"))

    swapped = flat.copy()
    swapped = cv2.rotate(swapped, cv2.ROTATE_180)              # camera rotated 180 degrees
    ids, critical, stable = run_pipeline(module, swapped)
    results.append(check('rotated image marker IDs', sorted(ids or []), required))
    # A 180-rotated camera view warps to the same top-down view rotated 180 degrees:
    # C4 -> (900-225, 900-300) = I8 and G8 -> (900-525, 900-600) = E4.
    results.append(check('rotated image critical cells', critical, ['I8']))
    results.append(check('rotated image stable cells', stable, ['E4']))

    print()
    print('ALL CHECKS PASSED' if all(results) else 'SOME CHECKS FAILED')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
