"""Convert a raw dataset into the canonical scene format.

    python scripts/prepare_data.py endonerf data/raw/endonerf/.../pulling_soft_tissues data/scenes/endonerf_pulling
    python scripts/prepare_data.py stereomis data/raw/stereomis/StereoMIS/P2_1 data/scenes/stereomis_P2_1 \
        --start 0 --stop 600 --stride 2 --width 640
"""
import argparse

from sr3d.data.converters import endonerf, stereomis


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["endonerf", "stereomis"])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--width", type=int, default=640, help="output width (stereomis only)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if a.dataset == "endonerf":
        out = endonerf.convert(a.src, a.dst, overwrite=a.overwrite)
    else:
        out = stereomis.convert(a.src, a.dst, a.start, a.stop, a.stride, a.width, overwrite=a.overwrite)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
