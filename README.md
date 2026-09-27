# Khojo Drone — Task 1A (survivor detection in arena image)

Team ID placeholder: `1234` — replace it with your **actual team ID** using:

```powershell
python tools/rename_team.py --team <YOUR_TEAM_ID>            # add --dry-run to preview
```

That one command renames the script, updates this README, drops the stale zip and
writes `KD_<YOUR_TEAM_ID>.zip` ready for upload.

## Files to submit

One file is graded. Everything else in this folder builds, exercises or documents it.

| File | Submit? | What it is |
|---|---|---|
| `pico_ws/src/swift_pico/scripts/KD_<TEAM_ID>_task1a.py` | **yes** | the program: reads the photo, writes `<image>_results.txt` next to it |
| `pico_ws/src/swift_pico/scripts/KD_<TEAM_ID>.zip` | **yes** — upload this | that one script, zipped for the Student portal; it contains nothing else |
| `README.md`, `images/`, `tools/` | no | this documentation, the pictures + results, and the local test tools |

The evaluator runs the script itself, on images it supplies:

```bash
python3 KD_<TEAM_ID>_task1a.py --image <arena>.jpg
```

so these properties have to hold (all already true, and checked by `selftest.py`):

* `--image` is a required `argparse` argument and no image name is hard-coded anywhere,
* an unreadable image prints `Error: Cannot load image: <path>` and exits with status 1,
* the run never blocks: no `cv2.imshow`, no `cv2.waitKey`, no GUI window, no keypress wait,
* the results file is `<input stem>_results.txt` written next to the input image,
  overwritten if it already exists, holding exactly the four lines from the brief
  (marker IDs, blank line, `Critical Survivors: `, `Stable Survivors: `),
* it imports only `cv2`, `numpy` and the standard library (`argparse`, `os`, `sys`).

Debug output goes to the terminal only — the results file never contains anything
else, and everything this folder keeps in `images/` is for local checking.

## Folder layout

```text
khojo drone/
├── README.md
├── images/                                  # every picture AND its output, side by side
│   ├── my_arena.png                         # the picture you supplied (no markers)
│   ├── my_arena_with_markers.png            # same picture with 80/85/90/95 added
│   ├── my_arena_with_markers_preview.png    # arena box + expected labels drawn on it
│   ├── my_arena_with_markers_results.txt    # what KD_1234_task1a.py reported for it
│   ├── image_1.jpg / image_1_results.txt    # smoke-test input + its output
│   ├── test_flat.jpg / test_tilted.jpg / test_varied.jpg   (+ *_results.txt each)
│   ├── missing_marker.jpg                   # sample with marker 95 erased (abort-path test)
│   └── test_images/                         # the 8 regression images + results from verify_test.py
├── tools/                                   # scripts only - no pictures live here
│   ├── generate_test_image.py              # configurable arena generator (grid, blobs, obstacles, tilt)
│   ├── verify_test.py                      # 8 end-to-end cases: generate -> run -> compare
│   ├── make_test_image.py                  # minimal smoke image for selftest.py
│   ├── selftest.py                         # 14 unit-ish checks (flat/tilted/rotated/missing ID)
│   ├── add_markers_to_image.py             # adds the 4 corner markers to any arena picture
│   └── rename_team.py                      # renames script/README/zip to the real team ID
└── pico_ws/src/swift_pico/scripts/
    ├── KD_1234_task1a.py                   # <-- the submission file
    └── KD_1234.zip                         # <-- the submission archive (script only)

Target layout on the robot (this tree mirrors it):

~/pico_ws/src/swift_pico/scripts/KD_1234_task1a.py
```

**Where things go:** every run writes `<image stem>_results.txt` *next to the image it
read*, and `add_markers_to_image.py` writes its output next to its input too — so
drop any new picture into `images/` and its output appears beside it.

## Marker IDs

The four arena corner markers must be **80, 85, 90 and 95** (`REQUIRED_MARKER_IDS`
in the script). If any one of them is not detected the program says which ID is
missing, reports `Error: Need all four required ArUco markers: 80, 85, 90, 95`
and exits with status 1 — it never guesses an arena from the wrong markers.
Any *other* markers in shot (including extra ones inside the arena) are ignored.

If the competition ever changes the IDs, no code edit is needed:

```bash
python3 pico_ws/src/swift_pico/scripts/KD_1234_task1a.py --image images/image_1.jpg --required-ids 80,85,90,95
```

## 1. Setup (on the Pico / Ubuntu machine)

```bash
sudo apt install python3-opencv python3-numpy
```

Verify the libraries:

```bash
python3 -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
```

## 2. Run

```bash
cd "$HOME/khojo drone"                        # project root, the folder that holds images/
```

```bash
python3 pico_ws/src/swift_pico/scripts/KD_1234_task1a.py --image images/image_1.jpg
```

Output is written next to the input image as `images/image_1_results.txt`:

```
Detected marker IDs: [80, 85, 90, 95]

Critical Survivors: C4
Stable Survivors: G8
```

## 3. Submit

```bash
cd ~/pico_ws/src/swift_pico/scripts/
zip KD_1234.zip KD_1234_task1a.py
```

Upload `KD_1234.zip` to the Student portal in the Task 1A slot.

## Windows equivalents (for local testing on this PC)

```powershell
cd 'C:\Users\varun\khojo drone'

# run the detector (the results file lands next to the image)
python pico_ws\src\swift_pico\scripts\KD_1234_task1a.py --image images\image_1.jpg

# rebuild the submission zip
cd pico_ws\src\swift_pico\scripts
Compress-Archive -Path KD_1234_task1a.py -DestinationPath KD_1234.zip -Force
```

## 4. Local smoke test (no arena needed)

`tools/make_test_image.py` renders a synthetic arena: a light-grey mat with a dark
boundary line, four `DICT_4X4_250` markers (**80, 85, 90, 95**) touching the arena
corners, plus a red (critical) and a yellow (stable) survivor blob at known grid
cells. It prints the expected answer so the detector can be checked without the
real course image.

```powershell
cd 'C:\Users\varun\khojo drone'

# 1. build a test image into images\
python tools\make_test_image.py --out images\image_1.jpg

# 2. run the detector on it (results land next to the image)
python pico_ws\src\swift_pico\scripts\KD_1234_task1a.py --image images\image_1.jpg

# 3. run the built-in checks (flat / tilted / rotated / missing marker / extra markers)
python tools\selftest.py
```

Last verification run of `tools/selftest.py` on this machine (OpenCV 5.0.0, Python 3.13.5):

```
PASS  flat image marker IDs: got [80, 85, 90, 95], expected [80, 85, 90, 95]
PASS  flat image critical cells: got ['C4'], expected ['C4']
PASS  flat image stable cells: got ['G8'], expected ['G8']
PASS  tilted image marker IDs: got [80, 85, 90, 95], expected [80, 85, 90, 95]
PASS  tilted image critical cells: got ['C4'], expected ['C4']
PASS  tilted image stable cells: got ['G8'], expected ['G8']
PASS  non-required markers ignored: got [80, 85, 90, 95], expected [80, 85, 90, 95]
PASS  6-marker image critical cells: got ['C4'], expected ['C4']
PASS  6-marker image stable cells: got ['G8'], expected ['G8']
PASS  missing ID 95 aborts: got (None, [], []), expected (None, [], [])
PASS  abort names only the missing ID: got Error: Missing required marker IDs: [95], expected Error: Missing required marker IDs: [95]
PASS  rotated image marker IDs: got [80, 85, 90, 95], expected [80, 85, 90, 95]
PASS  rotated image critical cells: got ['I8'], expected ['I8']
PASS  rotated image stable cells: got ['E4'], expected ['E4']

ALL CHECKS PASSED
```

Command-line check of the abort path (marker 95 blotted out of the generated image):

```powershell
cd 'C:\Users\varun\khojo drone'

# build an image with one corner marker missing
python tools\make_test_image.py --out images\missing_marker.jpg --erase-id 95

# it is already in the repo; run the detector on it
python pico_ws\src\swift_pico\scripts\KD_1234_task1a.py --image images\missing_marker.jpg
```

Output (no results file is written and the exit status is 1):

```
Error: Missing required marker IDs: [95]
Error: Need all four required ArUco markers: 80, 85, 90, 95
exit_code=1
```

`KD_1234_task1a.py` is stored with LF line endings (`\n`) so the file behaves
cleanly on Linux, and the zip contains only that one file.

Last verification run of `tools/selftest.py` on this machine (OpenCV 5.0.0, Python 3.13.5):

```
PASS  flat image marker IDs: got [80, 85, 90, 95], expected [80, 85, 90, 95]
PASS  flat image critical cells: got ['C4'], expected ['C4']
PASS  flat image stable cells: got ['G8'], expected ['G8']
PASS  tilted image marker IDs: got [80, 85, 90, 95], expected [80, 85, 90, 95]
PASS  tilted image critical cells: got ['C4'], expected ['C4']
PASS  tilted image stable cells: got ['G8'], expected ['G8']
PASS  non-required markers ignored: got [80, 85, 90, 95], expected [80, 85, 90, 95]
PASS  6-marker image critical cells: got ['C4'], expected ['C4']
PASS  6-marker image stable cells: got ['G8'], expected ['G8']
PASS  missing ID 95 aborts: got (None, [], []), expected (None, [], [])
PASS  abort names only the missing ID: got Error: Missing required marker IDs: [95], expected Error: Missing required marker IDs: [95]
PASS  rotated image marker IDs: got [80, 85, 90, 95], expected [80, 85, 90, 95]
PASS  rotated image critical cells: got ['I8'], expected ['I8']
PASS  rotated image stable cells: got ['E4'], expected ['E4']

ALL CHECKS PASSED
```

## 5. Running the pipeline on an arena picture of your own

`tools/add_markers_to_image.py` puts the four corner markers (**80, 85, 90, 95**)
onto a picture that has none, so `KD_1234_task1a.py` can be run on it:

* markers are pasted *outside* the arena on a dark plate inside an added canvas
  margin -- markers flush with the picture border are dropped by ArUco's
  `minDistanceToBorder`, which is how corner markers silently vanish,
* the corner of the marker image lands exactly on the arena corner, so the
  rectangle the pipeline crops is exactly the arena,
* the arena defaults to a 12 x 12 cell window of the picture's own grid. Exactly
  12 cells, because the warp to 900 x 900 then keeps the 75 px pitch that
  `map_to_grid` assumes; within that cell the window is shifted to the offset
  that puts the survivor blobs closest to labelled intersections,
* the tool then prints the answer it expects and decodes its own output again to
  prove the four markers really define the arena (exit status 1 if they do not).

```powershell
cd 'C:\Users\varun\khojo drone'

# put the picture in images\, then add the markers (output + results land beside it)
python tools\add_markers_to_image.py --image images\my_arena.png --preview

# name the arena rectangle yourself instead of using the detected grid
python tools\add_markers_to_image.py --image images\my_arena.png --arena 160,156,1176,1117

# run the submission script on the result
python pico_ws\src\swift_pico\scripts\KD_1234_task1a.py --image images\my_arena_with_markers.png
```

Run on `images/my_arena.png` (1254 x 1254 picture with its own 13 x 14 cell grid:
14 grid columns and 15 grid rows, five survivors, black obstacles, green foliage,
blue floor). Everything below is one run, in order:

```
Input: images\my_arena.png (1254x1254)
Survivor blobs in the picture: critical=2, stable=3
Picture grid detected: 14 columns, 15 rows
Arena: (128, 180)-(1144, 1141) (auto: 12 x 12 grid cells with 5 survivor blob(s) inside, worst blob 0.27 cells from an intersection)
Wrote images\my_arena_with_markers.png (1558x1558, 152 px margin per side)
Marker check (same detector settings as the pipeline):
  decoded marker IDs: [80, 85, 90, 95]
  marker 90: arena corner (1296, 1293) off by 0.0 px -> ok
  marker 95: arena corner (280, 1293) off by 1.0 px -> ok
  marker 85: arena corner (1296, 332) off by 1.0 px -> ok
  marker 80: arena corner (280, 332) off by 1.4 px -> ok
Survivor labels the pipeline should report:
  critical blob at (301, 959) -> B10, 0.27 cells from the nearest labelled intersection
  critical blob at (963, 197) -> J1, 0.21 cells from the nearest labelled intersection
  stable blob at (541, 886) -> E9, 0.18 cells from the nearest labelled intersection
  stable blob at (964, 590) -> J5, 0.13 cells from the nearest labelled intersection
  stable blob at (455, 270) -> D1, 0.13 cells from the nearest labelled intersection
Expected results file:
  Detected marker IDs: [80, 85, 90, 95]
  Critical Survivors: B10, J1
  Stable Survivors: E9, J5, D1
Wrote images\my_arena_with_markers_preview.png (arena, blobs, expected labels)
```

`KD_1234_task1a.py` produced exactly that predicted answer:

```
Detected marker IDs: [80, 85, 90, 95]

Critical Survivors: B10, J1
Stable Survivors: E9, J5, D1
```

Three things to know when the picture is not a real arena:

* the labels come from the markers, not from the picture. This picture's grid has
  13 x 14 big cells and its objects sit in the *middle* of those cells, while the
  pipeline labels grid *intersections* -- so the tool picks the 12-cell window
  and offset with the most slack (0.27 cells here, about 20 px). A blob sitting
  exactly halfway between two intersections is always on the rounding boundary
  and its label is a coin flip; that is a property of the picture, not the code.
* the HSV colour bounds are the lighting-sensitive part. Here they picked up
  exactly the 2 red triangles and the 3 yellow circles and ignored the blue
  floor, the white grid, the green foliage and the black obstacles, so no tuning
  was needed.
* `tools/generate_test_image.py` still gives the only images whose labels are
  guaranteed correct, because there the arena *is* drawn as 12 x 12 cells with
  the survivors painted on labelled intersections.

## Notes about the algorithm

* `detect_markers` uses the modern `cv2.aruco.ArucoDetector` API (OpenCV >= 4.7;
  verified locally on OpenCV 5.0.0).
* `select_corner_markers` only accepts the required IDs (`REQUIRED_MARKER_IDS`,
  default 80/85/90/95, overridable with `--required-ids`). If any required ID is
  missing it prints which ones and returns `None`, so `main` exits with status 1
  instead of warping a wrong arena. Extra markers in the frame are ignored.
* `get_arena_corners` uses, for each corner marker, the corner point closest to
  the centroid of all marker centres — the marker corner that touches the arena
  — and sorts them with `order_corner_points` into TL, TR, BR, BL. Because the
  sort is purely geometric, the script also works when the camera is rotated.
* `perspective_transform` warps that rectangle to a 900x900 top-down view.
* `detect_survivors` thresholds red (two HSV ranges, critical) and yellow
  (stable), cleans the masks with a 5x5 morphological open, and drops contours
  smaller than 100 px to reject noise. If the real arena prints look washed out
  or dark, tune these HSV bounds — that is the only lighting-sensitive part.
* `map_to_grid` converts pixel centres to labels using a 75 px cell pitch
  starting at 75 px, clamped to A..K and 1..11 (11x11 grid).
* Output `Detected marker IDs:` is sorted; the survivor labels are in detection
  order left-to-right per mask (order within a line is not guaranteed).
