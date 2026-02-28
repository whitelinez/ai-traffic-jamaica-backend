"""
delete_bucket.py — Deletes all objects in the ml-datasets bucket using the Supabase Storage REST API.
Run from the backend directory. Reads credentials from environment or .env file.
"""
import os
import sys
import requests

# ── Load .env if present ────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on shell env

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SERVICE_KEY  = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
BUCKET       = "ml-datasets"
DRY_RUN      = "--dry-run" in sys.argv

if not SUPABASE_URL or not SERVICE_KEY:
    sys.exit("[ERROR] SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.")

HEADERS = {
    "apikey":        SERVICE_KEY,
    "Authorization": f"Bearer {SERVICE_KEY}",
    "Content-Type":  "application/json",
}

def list_objects(prefix: str, limit: int = 1000, offset: int = 0) -> list[dict]:
    url  = f"{SUPABASE_URL}/storage/v1/object/list/{BUCKET}"
    body = {"prefix": prefix, "limit": limit, "offset": offset, "sortBy": {"column": "name", "order": "asc"}}
    r = requests.post(url, json=body, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

def delete_objects(names: list[str]) -> dict:
    url  = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}"
    body = {"prefixes": names}
    r = requests.delete(url, json=body, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()

def collect_all(prefix: str) -> list[str]:
    """Recursively collect every file path under a prefix."""
    paths: list[str] = []
    offset = 0
    while True:
        items = list_objects(prefix, limit=1000, offset=offset)
        if not items:
            break
        for item in items:
            name = item.get("name", "")
            full = f"{prefix}/{name}" if prefix else name
            if item.get("id"):        # it's a file
                paths.append(full)
            else:                      # it's a folder — recurse
                paths.extend(collect_all(full))
        if len(items) < 1000:
            break
        offset += 1000
    return paths

def chunked(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# ── Main ─────────────────────────────────────────────────────────────────────
print(f"[delete_bucket] Scanning bucket: {BUCKET}")
if DRY_RUN:
    print("[delete_bucket] DRY RUN — no files will be deleted")

all_files = collect_all("")
print(f"[delete_bucket] Found {len(all_files)} objects")

if not all_files:
    print("[delete_bucket] Nothing to delete.")
    sys.exit(0)

# Print first 10 for confirmation
print("\nSample files:")
for f in all_files[:10]:
    print(f"  {f}")
if len(all_files) > 10:
    print(f"  ... and {len(all_files) - 10} more")

if not DRY_RUN:
    confirm = input(f"\nDelete ALL {len(all_files)} files from '{BUCKET}'? [yes/N]: ").strip()
    if confirm.lower() != "yes":
        print("Aborted.")
        sys.exit(0)

    deleted = 0
    for batch in chunked(all_files, 100):
        result = delete_objects(batch)
        deleted += len(batch)
        print(f"  Deleted {deleted}/{len(all_files)}...")

    print(f"\n[delete_bucket] Done. {deleted} objects removed from '{BUCKET}'.")
else:
    print("\n[DRY RUN] No files deleted. Remove --dry-run to execute.")
