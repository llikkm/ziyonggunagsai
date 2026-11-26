import hashlib
from pathlib import Path


HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
FROZEN={'03_laser_triangulation/laser_lib.py':'f20e34d7f5760d03adc3a14356ad382f7c99d8a5',
        '06_standard_scan/replay.py':'4223e4143fe7328aedd7ffbd22c8522ae031bf73',
        '06_standard_scan/simulate.py':'def5066cd1a190267d6ead3d9627749d9048cc1f',
        '07_heightmap_volume/estimator.py':'76ed16886a71d58ad6b352aedc6f5b874a3420a1',
        '09_volume_refinement/refined.py':'dad0b69aa99a14aeb476ca783c9ede992392b2b7'}
POSES=[(0,0,0),(7,-5,24),(-9,7,53)]


def check_lock():
    for path,digest in FROZEN.items():
        data=(ROOT/path).read_bytes()
        if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=digest:
            raise ValueError(f'Frozen source changed: {path}')
