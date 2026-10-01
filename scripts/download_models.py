"""Downloads the MediaPipe Face Landmarker model (~3.6 MB) into models/.

Needed only with newer MediaPipe versions (0.10.22+), which dropped the legacy
`mp.solutions` API. Run once on the Pi (or any machine with internet):
    python scripts/download_models.py
"""
import urllib.request
from pathlib import Path

URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
       "face_landmarker/float16/1/face_landmarker.task")
dest = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
dest.parent.mkdir(exist_ok=True)
if dest.exists():
    print(f"Already have {dest}")
else:
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, dest)
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
