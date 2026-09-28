#!/usr/bin/env python3
"""End-to-end check: generate test images, run KD_*_task1a.py, compare with expectations.

Usage:
    python tools/verify_test.py                 # uses pico_ws/.../scripts/KD_*_task1a.py
    python tools/verify_test.py path/to/KD_4373_task1a.py

Images land in images/test_images/ with their <image>_results.txt.  Label order
follows contour order, so labels are compared as sorted sets.
"""
import ast
import glob
import os
import subprocess
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
SCRIPTS_DIR = os.path.join(ROOT, 'pico_ws', 'src', 'swift_pico', 'scripts')
IMAGE_DIR = os.path.join(ROOT, 'images', 'test_images')
GENERATOR = os.path.join(TOOLS_DIR, 'generate_test_image.py')

MARKERS = [80, 85, 90, 95]

TEST_CASES = [
    {
        'name': 'Flat basic',
        'args': ['--critical', 'C4,F2', '--stable', 'G8,D6', '--no-perspective'],
        'expected': {'markers': MARKERS, 'critical': ['C4', 'F2'], 'stable': ['D6', 'G8']},
    },
    {
        'name': 'Perspective basic',
        'args': ['--critical', 'C4,F2', '--stable', 'G8,D6'],
        'expected': {'markers': MARKERS, 'critical': ['C4', 'F2'], 'stable': ['D6', 'G8']},
    },
    {
        'name': 'Flat edges',
        'args': ['--critical', 'A1,K11', '--stable', 'F6,B3', '--no-perspective'],
        'expected': {'markers': MARKERS, 'critical': ['A1', 'K11'], 'stable': ['B3', 'F6']},
    },
    {
        'name': 'Perspective edges',
        'args': ['--critical', 'A1,K11', '--stable', 'F6,B3'],
        'expected': {'markers': MARKERS, 'critical': ['A1', 'K11'], 'stable': ['B3', 'F6']},
    },
    {
        'name': 'Many survivors flat',
        'args': ['--critical', 'C4,G8,K11,A1', '--stable', 'F6,B3,D6,H2', '--no-perspective'],
        'expected': {'markers': MARKERS, 'critical': ['A1', 'C4', 'G8', 'K11'],
                     'stable': ['B3', 'D6', 'F6', 'H2']},
    },
    {
        'name': 'Many survivors perspective',
        'args': ['--critical', 'C4,G8,K11,A1', '--stable', 'F6,B3,D6,H2'],
        'expected': {'markers': MARKERS, 'critical': ['A1', 'C4', 'G8', 'K11'],
                     'stable': ['B3', 'D6', 'F6', 'H2']},
    },
    {
        'name': 'Seed 7 obstacles',
        'args': ['--critical', 'C4,F2', '--stable', 'G8,D6', '--seed', '7'],
        'expected': {'markers': MARKERS, 'critical': ['C4', 'F2'], 'stable': ['D6', 'G8']},
    },
    {
        'name': 'Single survivor per class',
        'args': ['--critical', 'K1', '--stable', 'A11'],
        'expected': {'markers': MARKERS, 'critical': ['K1'], 'stable': ['A11']},
    },
]


def find_submission(explicit=None):
    if explicit:
        if not os.path.exists(explicit):
            raise SystemExit(f'Error: no such script: {explicit}')
        return os.path.abspath(explicit)
    matches = glob.glob(os.path.join(SCRIPTS_DIR, 'KD_*_task1a.py'))
    if len(matches) != 1:
        raise SystemExit(f'Error: pass the script explicitly, found {matches}')
    return matches[0]


def parse_results(text):
    """Read the three result lines into comparable values."""
    parsed = {'markers': [], 'critical': [], 'stable': []}
    for line in text.splitlines():
        if line.startswith('Detected marker IDs:'):
            parsed['markers'] = sorted(int(v) for v in
                                       ast.literal_eval(line.split(':', 1)[1].strip()))
        elif line.startswith('Critical Survivors:'):
            parsed['critical'] = sorted(part.strip() for part in
                                        line.split(':', 1)[1].split(',') if part.strip())
        elif line.startswith('Stable Survivors:'):
            parsed['stable'] = sorted(part.strip() for part in
                                      line.split(':', 1)[1].split(',') if part.strip())
    return parsed


def run_case(case, script):
    slug = case['name'].lower().replace(' ', '_')
    image_path = os.path.join(IMAGE_DIR, f'test_{slug}.jpg')
    results_path = os.path.splitext(image_path)[0] + '_results.txt'

    generate = subprocess.run([sys.executable, GENERATOR, '--out', image_path] + case['args'],
                              capture_output=True, text=True)
    if generate.returncode != 0:
        return False, f'generator failed:\n{generate.stdout}{generate.stderr}', image_path

    if os.path.exists(results_path):
        os.remove(results_path)

    run = subprocess.run([sys.executable, script, '--image', image_path],
                         capture_output=True, text=True)
    if run.returncode != 0:
        return False, f'pipeline exited {run.returncode}:\n{run.stdout}{run.stderr}', image_path
    if not os.path.exists(results_path):
        return False, 'no results file was written', image_path

    with open(results_path, 'r', encoding='utf-8') as handle:
        text = handle.read()

    actual = parse_results(text)
    problems = []
    for key in ('markers', 'critical', 'stable'):
        if actual[key] != case['expected'][key]:
            problems.append(f"{key}: got {actual[key]}, expected {case['expected'][key]}")

    if problems:
        return False, '\n'.join(problems) + f'\nresults file was:\n{text.strip()}', image_path
    return True, (f"markers {actual['markers']} | critical {actual['critical']} | "
                  f"stable {actual['stable']}"), image_path


def main():
    script = find_submission(sys.argv[1] if len(sys.argv) > 1 else None)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    print(f'submission : {os.path.relpath(script, ROOT)}')
    print(f'images     : {os.path.relpath(IMAGE_DIR, ROOT)}')
    print(f'cases      : {len(TEST_CASES)}')

    passed = 0
    for index, case in enumerate(TEST_CASES, start=1):
        print(f"\n{'=' * 62}\nTEST {index}/{len(TEST_CASES)}: {case['name']}  "
              f"({' '.join(case['args'])})\n{'=' * 62}")
        ok, message, image_path = run_case(case, script)
        print(f"{'PASS' if ok else 'FAIL'}: {message}")
        print(f'image : {os.path.relpath(image_path, ROOT)}')
        passed += 1 if ok else 0

    print(f"\n{'=' * 62}\nRESULT: {passed}/{len(TEST_CASES)} tests passed\n{'=' * 62}")
    return 0 if passed == len(TEST_CASES) else 1


if __name__ == '__main__':
    sys.exit(main())
