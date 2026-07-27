"""list_drive.py — dump a drive's folder/file tree to a text file.

Usage (Windows):
    python list_drive.py F:\\HM_neurons                 # folders only, depth 6
    python list_drive.py F:\\HM_neurons --files --depth 4   # + file names/sizes
    python list_drive.py F:\\ --out drive.txt           # whole drive

Writes an indented tree (with per-folder file counts, and sizes if --files)
to <root>_listing.txt, skipping folders it can't read.
"""
import os
import sys
import argparse


def human(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder/drive to list, e.g. F:\\HM_neurons")
    ap.add_argument("--out", default=None, help="output text file")
    ap.add_argument("--depth", type=int, default=6, help="max folder depth (default 6)")
    ap.add_argument("--files", action="store_true", help="also list files + sizes (bigger output)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out = args.out or (root.rstrip("\\/").replace(":", "").replace("\\", "_")
                       .replace("/", "_") + "_listing.txt")
    n_dirs = n_files = 0
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# listing of {root}\n\n")
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
            depth = dirpath[len(root):].count(os.sep)
            dirnames.sort()
            filenames.sort()
            indent = "  " * depth
            f.write(f"{indent}{os.path.basename(dirpath) or dirpath}\\  "
                    f"({len(filenames)} files)\n")
            n_dirs += 1
            if args.files:
                for name in filenames:
                    try:
                        sz = os.path.getsize(os.path.join(dirpath, name))
                    except OSError:
                        sz = 0
                    f.write(f"{indent}  {name}    {human(sz)}\n")
                    n_files += 1
            if depth >= args.depth:
                dirnames[:] = []          # don't descend deeper
    print(f"wrote {out}: {n_dirs} folders, {n_files} files")


if __name__ == "__main__":
    main()
