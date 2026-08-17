"""make_channel_map.py — (re)build the LFP channel_map.npy on its own.

`channel_map.npy` records, for each COLUMN of `lfp_data.npy`, which Trodes nTrode /
channel it is: a list of dicts ``{'index', 'ntrode', 'channel', 'source_file'}``.
`export_lfp.py` writes it as a side output while exporting the LFP; this standalone
rebuilds the SAME file straight from the Trodes
``<recording>.LFP/<recording>.LFP_nt<N>ch<C>.dat`` filenames — WITHOUT loading any
voltage data — in the exact column order `export_lfp.py` uses:

  the channels common to every concatenated session, sorted by ntrode then channel
  (any files whose name has no ``_nt<N>ch<C>`` get ntrode/channel = None and sort
  last), with ``source_file`` taken from the first session.

So a `channel_map.npy` produced here lines up 1:1 with the columns of an existing
`lfp_data.npy` (a sanity check against it is printed when it is present).

Needs the Trodes `.LFP` export (the nTrode/channel identity lives only in the .dat
filenames) — it cannot be recovered from `lfp_data.npy` alone.

Usage:
    python make_channel_map.py --input_folder <trodes_export_dir> [--output_folder <dir>]
"""
import re
import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from session_prefix import session_prefix   # noqa: E402  (Rat6_20260707_091045_ prefix)


_NT_CH_RE = re.compile(r"_nt(\d+)ch(\d+)")


def parse_channel_info(filename):
    """(ntrode, channel) from a Trodes LFP filename, e.g. '...LFP_nt3ch1.dat' ->
    (3, 1); (None, None) if the name has no _nt<N>ch<C> token."""
    m = _NT_CH_RE.search(Path(filename).stem)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _sort_key(f):
    nt, ch = parse_channel_info(f)
    return (nt is None, nt or 0, ch or 0)          # unparsed ntrodes sort last


def find_lfp_sessions(input_folder):
    """[{'name', 'lfp_files'}] — one per '<recording>.LFP' folder (each .rec session),
    in chronological (name) order; falls back to flat '*LFP*.dat' as one session.
    Only the .dat filenames are read, never their contents."""
    base = Path(input_folder)
    sessions = []
    for d in sorted((d for d in base.glob("*.LFP") if d.is_dir()), key=lambda d: d.name):
        files = sorted((f for f in d.glob("*.dat") if "timestamps" not in f.name.lower()),
                       key=_sort_key)
        if files:
            sessions.append({"name": d.stem, "lfp_files": files})
    if not sessions:
        flat = sorted((f for f in base.glob("*LFP*.dat") if "timestamps" not in f.name.lower()),
                      key=_sort_key)
        if flat:
            sessions.append({"name": base.name, "lfp_files": flat})
    return sessions


def build_channel_map(input_folder):
    """(prefix, sessions, channel_map_list). channel_map_list matches lfp_data.npy's
    columns: the (ntrode, channel) common to every session, sorted nt-then-ch, with
    source_file from the first session. Empty list if no LFP .dat files found."""
    sessions = find_lfp_sessions(input_folder)
    if not sessions:
        return "", [], []
    fmaps = [{parse_channel_info(f): f for f in s["lfp_files"]} for s in sessions]
    common = set(fmaps[0])
    for fm in fmaps[1:]:
        common &= set(fm)
    keys = sorted(common, key=lambda k: (k[0] is None, k[0] or 0, k[1] or 0))
    ch_map = [{"index": i, "ntrode": k[0], "channel": k[1],
               "source_file": fmaps[0][k].name} for i, k in enumerate(keys)]
    return session_prefix(sessions[0]["name"]), sessions, ch_map


def run(input_folder, output_folder=None):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder) if output_folder else input_folder
    pfx, sessions, ch_map = build_channel_map(input_folder)
    if not ch_map:
        print(f"No Trodes LFP .dat files under {input_folder}\n"
              f"  (expected <recording>.LFP/<recording>.LFP_nt<N>ch<C>.dat, or flat *LFP*.dat).")
        return None
    output_folder.mkdir(parents=True, exist_ok=True)
    out = output_folder / f"{pfx}channel_map.npy"
    np.save(out, ch_map)

    n_none = sum(1 for c in ch_map if c["ntrode"] is None)
    print(f"{len(sessions)} session(s); {len(ch_map)} channel(s) common to all.")
    print(f"  column order: nt{ch_map[0]['ntrode']}ch{ch_map[0]['channel']} .. "
          f"nt{ch_map[-1]['ntrode']}ch{ch_map[-1]['channel']}"
          + (f"   ({n_none} with no ntrode)" if n_none else ""))
    # sanity check against an existing lfp_data.npy (same prefix), if present
    ld = output_folder / f"{pfx}lfp_data.npy"
    if ld.exists():
        try:
            arr = np.load(ld, mmap_mode="r")
            ncol = arr.shape[1] if arr.ndim == 2 else 1
            print(f"  lfp_data.npy has {ncol} columns vs {len(ch_map)} map entries: "
                  f"{'OK' if ncol == len(ch_map) else 'MISMATCH — check the export!'}")
        except Exception as e:
            print(f"  (could not check lfp_data.npy: {e})")
    print(f"  wrote {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Rebuild the LFP channel_map.npy from Trodes .LFP exports (no data load).")
    ap.add_argument("--input_folder", required=True,
                    help="folder holding <recording>.LFP/*.dat (or flat *LFP*.dat).")
    ap.add_argument("--output_folder", default=None,
                    help="where to write channel_map.npy (default: the input folder).")
    ap.add_argument("--config", default=None, help="Accepted for runner consistency (unused).")
    args = ap.parse_args()
    run(args.input_folder, args.output_folder)
