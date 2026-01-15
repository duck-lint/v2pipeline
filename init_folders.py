from pathlib import Path
import argparse
from common import ensure_dir

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Project root")
    args = ap.parse_args()

    print(f"[stage_folders] args: {args}")

    root = Path(args.root).resolve()
    ensure_dir(root / "stage_0_raw")
    ensure_dir(root / "stage_1_clean")
    ensure_dir(root / "stage_2_chunks")
    ensure_dir(root / "stage_3_chroma")

    print("Created/verified folders:")
    print(f" - {root / 'stage_0_raw'}")
    print(f" - {root / 'stage_1_clean'}")
    print(f" - {root / 'stage_2_chunks'}")
    print(f" - {root / 'stage_3_chroma'}")

if __name__ == "__main__":
    main()
