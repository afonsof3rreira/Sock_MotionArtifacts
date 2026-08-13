import os
import re
import sys

IDENTITY_FILENAME = "id.txt"

def load_identity(root_path):
    path = os.path.join(root_path, IDENTITY_FILENAME)
    if not os.path.isfile(path):
        sys.exit(
            f"Missing '{IDENTITY_FILENAME}' in {os.path.abspath(root_path)}.\n"
            "This file should have been sent to you — copy it into "
            "the Annotator ID folder before starting. Do not create it yourself."
        )

    with open(path, encoding="utf-8-sig") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) < 2:
        sys.exit(f"'{IDENTITY_FILENAME}' must have the name on line 1 and the ID on line 2.")

    name, annotator_id = lines[0], lines[1]
    if not re.fullmatch(r"A\d{1,3}", annotator_id):
        sys.exit(f"Malformed annotator ID '{annotator_id}' in '{IDENTITY_FILENAME}'.")

    return name, annotator_id