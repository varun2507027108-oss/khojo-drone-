#!/usr/bin/env python3
"""Rename the Task 1A script to the real team ID and rebuild the submission zip.

Usage:
    python tools/rename_team.py --team 4373             # rename + rebuild zip
    python tools/rename_team.py --team 4373 --dry-run   # only report what it would do

Renames KD_<old>_task1a.py, points the team ID references in README.md at the new
ID, deletes the stale zip and writes KD_<team>.zip holding only the script.
"""
import argparse
import glob
import os
import re

import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.join(ROOT, 'pico_ws', 'src', 'swift_pico', 'scripts')
README_PATH = os.path.join(ROOT, 'README.md')
SUFFIX = '_task1a.py'


def find_script():
    matches = glob.glob(os.path.join(SCRIPT_DIR, 'KD_*' + SUFFIX))
    if not matches:
        raise SystemExit(f'Error: no KD_*{SUFFIX} found in {SCRIPT_DIR}')
    if len(matches) > 1:
        names = ', '.join(os.path.basename(path) for path in matches)
        raise SystemExit(f'Error: expected one script, found several: {names}')
    return matches[0]


def team_id_from(script_path):
    return os.path.basename(script_path)[len('KD_'):-len(SUFFIX)]


def update_readme(old_id, new_id, dry_run):
    """Point every team-ID occurrence in README.md at new_id (even if it drifted)."""
    if not os.path.exists(README_PATH):
        return False
    with open(README_PATH, 'r', encoding='utf-8') as handle:
        text = handle.read()

    updated = re.sub(r'KD_([A-Za-z0-9]+)(_task1a\.py|\.zip)',
                     lambda match: f'KD_{new_id}{match.group(2)}', text)
    updated = re.sub(r'(Team ID placeholder:\s*`)([A-Za-z0-9]+)(`)',
                     lambda match: match.group(1) + new_id + match.group(3), updated)

    if updated == text:
        return False
    if not dry_run:
        with open(README_PATH, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(updated)
    return True


def build_zip(script_path, team_id, dry_run):
    zip_path = os.path.join(SCRIPT_DIR, f'KD_{team_id}.zip')
    if dry_run:
        return zip_path
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(script_path, arcname=os.path.basename(script_path))
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise SystemExit(f'Error: corrupt entry in {zip_path}: {bad}')
    return zip_path


def main():
    parser = argparse.ArgumentParser(description='Rename Task 1A files to the real team ID')
    parser.add_argument('--team', required=True, help='Your team ID, e.g. 4373')
    parser.add_argument('--dry-run', action='store_true', help='Report actions without changing files')
    args = parser.parse_args()

    team_id = args.team.strip()
    if not team_id or not team_id.replace('_', '').isalnum():
        raise SystemExit(f'Error: --team must be alphanumeric, got: {args.team}')

    current_script = find_script()
    old_id = team_id_from(current_script)
    new_script = os.path.join(SCRIPT_DIR, f'KD_{team_id}{SUFFIX}')
    old_zip = os.path.join(SCRIPT_DIR, f'KD_{old_id}.zip')

    print(f"team ID : {old_id} -> {team_id}")
    print(f"script  : {os.path.basename(current_script)} -> {os.path.basename(new_script)}")

    if args.dry_run:
        print(f"readme  : would point KD_* names and the placeholder line at '{team_id}'")
        print(f"zip     : would delete {os.path.basename(old_zip) if os.path.exists(old_zip) else '(none)'} "
              f"and write {os.path.basename(build_zip(new_script, team_id, True))}")
        print('dry run: nothing was changed')
        return 0

    if old_id != team_id:
        os.replace(current_script, new_script)
        if os.path.exists(old_zip):
            os.remove(old_zip)

    if update_readme(old_id, team_id, dry_run=False):
        print(f"readme  : team ID references updated to {team_id}")
    zip_path = build_zip(new_script, team_id, dry_run=False)

    print(f"zip     : {os.path.basename(zip_path)} ({os.path.getsize(zip_path)} bytes, contains "
          f"{os.path.basename(new_script)})")
    print('done')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
