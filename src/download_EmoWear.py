from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Optional

import requests
from tqdm import tqdm

# EmoWear Zenodo record
RECORD_ID = "10407279"
BASE_URL = f"https://zenodo.org/records/{RECORD_ID}/files"

# Files listed on the Zenodo page (names + md5 from the record page)
EMOWEAR_FILES: Dict[str, Optional[str]] = {
    "csv.zip": "8157ec5e81c1c979d4cd2614bd4c3f89",
    # "mat.zip": "4a62f4292659b03764d388ff4a615628",
    # "meta.csv": "d5dfd9a877b13fda3f3e1a24e1fe3fdf",
    "questionnaire.csv": "67dfc112a3eb199a6be506ac64797d1e",
    # "raw.zip": "20a8d894c05472e487af5d8bb0c59599",
    # "sample.zip": "e0b3fa12da1b814c80f26678b5dc3592",
}


def md5_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(
    filename: str,
    out_dir: str = "emowear_downloads",
    verify_md5: bool = True,
    timeout: int = 30,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """
    Download a single file from the EmoWear Zenodo record.
    Supports resume if a partial file already exists.
    """
    if filename not in EMOWEAR_FILES:
        raise ValueError(
            f"Unknown file '{filename}'. Available: {', '.join(EMOWEAR_FILES.keys())}"
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / filename

    url = f"{BASE_URL}/{filename}?download=1"

    # Resume support
    existing_size = file_path.stat().st_size if file_path.exists() else 0
    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    with requests.Session() as session:
        # stream=True to avoid loading large files in memory
        with session.get(url, stream=True, headers=headers, timeout=timeout) as r:
            # If server doesn't honor Range, it may return 200 instead of 206
            if r.status_code not in (200, 206):
                raise RuntimeError(f"Failed to download {filename}: HTTP {r.status_code}")

            # Determine total size for progress bar
            if r.status_code == 206:
                # Partial content; content-length is remaining bytes
                remaining = int(r.headers.get("Content-Length", "0"))
                total_size = existing_size + remaining
                mode = "ab"
            else:
                # Fresh download (or server ignored Range)
                total_size = int(r.headers.get("Content-Length", "0")) or None
                mode = "wb"
                if existing_size > 0:
                    print(f"[WARN] Server did not resume {filename}; restarting download.")
                    existing_size = 0

            with file_path.open(mode) as f, tqdm(
                total=total_size,
                initial=existing_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter keep-alive chunks
                        f.write(chunk)
                        pbar.update(len(chunk))

    # Optional MD5 verification
    expected_md5 = EMOWEAR_FILES[filename]
    if verify_md5 and expected_md5:
        actual_md5 = md5_of_file(file_path)
        if actual_md5.lower() != expected_md5.lower():
            raise RuntimeError(
                f"MD5 mismatch for {filename}\n"
                f"Expected: {expected_md5}\n"
                f"Actual:   {actual_md5}"
            )
        print(f"[OK] MD5 verified for {filename}")

    return file_path


def download_emowear(
    files: Optional[list[str]] = None,
    out_dir: str = "EmoWear",
    verify_md5: bool = True,
) -> None:
    """
    Download selected EmoWear files (or all files if files=None).
    """
    if files is None:
        files = list(EMOWEAR_FILES.keys())

    for fn in files:
        print(f"\nDownloading: {fn}")
        try:
            path = download_file(fn, out_dir=out_dir, verify_md5=verify_md5)
            print(f"Saved to: {path}")
        except Exception as e:
            print(f"[ERROR] {fn}: {e}")


if __name__ == "__main__":
    # Example 1: Download only the smaller sample package first
    download_emowear(out_dir="EmoWear")