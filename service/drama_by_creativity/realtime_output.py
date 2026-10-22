import os
import json
from drama_local.runtime import atomic_json


def save_realtime(path_parts: list, data):
    base = os.environ.get("DRAMA_OUTPUT_DIR")
    if not base:
        return
    *dirs, filename = path_parts
    full_dir = os.path.join(base, *dirs)
    os.makedirs(full_dir, exist_ok=True)
    filepath = os.path.join(full_dir, filename)
    atomic_json(filepath, json.loads(json.dumps(data, ensure_ascii=False, default=str)))
