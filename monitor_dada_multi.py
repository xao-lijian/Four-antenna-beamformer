#!/usr/bin/env python3
"""
Monitor statistics of multiple PSRDADA ringbuffers.

Default keys:
  0xa000 0xa002 0xa004 0xa006 0xb000

Example:
  python monitor_dada_multi.py
  python monitor_dada_multi.py --interval 0.5 --count 0 --clear
  python monitor_dada_multi.py --keys 0xa000 0xb000 --reader 0
"""

import argparse
import time
from typing import Optional, Tuple, Dict, List

from psrdada import Viewer


def parse_key(s: str) -> int:
    s = s.strip().lower()
    if s.startswith("0x"):
        return int(s, 16)
    # allow "a000" style
    if all(c in "0123456789abcdef" for c in s) and len(s) <= 8:
        return int(s, 16)
    return int(s, 0)


def try_open_viewer(key: int) -> Optional[Viewer]:
    try:
        return Viewer(key)
    except Exception:
        return None


def try_get_stats(viewer: Viewer, reader_idx: int) -> Tuple[Optional[Dict], Optional[Dict], Optional[int]]:
    """
    Returns (hdr_mon, data_mon, n_readers). Any of them can be None on failure.
    """
    try:
        hdr_mon, data_mon_list = viewer.getbufstats()
        n_readers = len(data_mon_list) if data_mon_list is not None else 0

        data_mon = None
        if data_mon_list and 0 <= reader_idx < len(data_mon_list):
            data_mon = data_mon_list[reader_idx]

        return hdr_mon, data_mon, n_readers
    except Exception:
        return None, None, None


def fmt_stats(mon: Optional[Dict], w_w: int = 8, w_r: int = 8) -> str:
    if not mon:
        return f"{'NA':>6} {'NA':>6} {'NA':>6} {'NA':>{w_w}} {'NA':>{w_r}}"
    return f"{mon.get('FREE', -1):6d} {mon.get('FULL', -1):6d} {mon.get('CLEAR', -1):6d} {mon.get('NWRITE', -1):{w_w}d} {mon.get('NREAD', -1):{w_r}d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--keys",
        nargs="+",
        default=["0xa000", "0xa002", "0xa004", "0xa006", "0xb000"],
        help="DADA keys (hex like 0xa000). Default: 0xa000 0xa002 0xa004 0xa006 0xb000",
    )
    ap.add_argument("--interval", type=float, default=1.0, help="Refresh interval seconds (default 1.0)")
    ap.add_argument("--count", type=int, default=0, help="How many iterations to run. 0 means infinite (default 0)")
    ap.add_argument("--reader", type=int, default=0, help="Which reader index to display for DATA stats (default 0)")
    ap.add_argument("--clear", action="store_true", help="Clear screen each refresh")
    args = ap.parse_args()

    keys = [parse_key(k) for k in args.keys]

    viewers: Dict[int, Optional[Viewer]] = {k: try_open_viewer(k) for k in keys}

    header = (
        f"{'TIME':<19} {'KEY':>8} {'READERS':>7} | "
        f"{'H_FREE':>6} {'H_FULL':>6} {'H_CLR':>6} {'H_W':>8} {'H_R':>8} || "
        f"{'D_FREE':>6} {'D_FULL':>6} {'D_CLR':>6} {'D_W':>8} {'D_R':>8}"
    )

    i = 0
    try:
        while True:
            if args.count > 0 and i >= args.count:
                break
            i += 1

            if args.clear:
                print("\033[2J\033[H", end="")  # ANSI clear + home

            print(header)
            print("-" * len(header))

            now = time.strftime("%Y-%m-%d %H:%M:%S")

            for k in keys:
                v = viewers.get(k)

                # If not opened yet, try reconnect
                if v is None:
                    v = try_open_viewer(k)
                    viewers[k] = v

                if v is None:
                    # still not available
                    print(f"{now:<19} {hex(k):>8} {'-':>7} | "
                          f"{'DISCONN':>6} {'':>6} {'':>6} {'':>8} {'':>8} || "
                          f"{'DISCONN':>6} {'':>6} {'':>6} {'':>8} {'':>8}")
                    continue

                hdr_mon, data_mon, n_readers = try_get_stats(v, args.reader)

                # If viewer got broken, drop it and show NA this round
                if hdr_mon is None and data_mon is None and n_readers is None:
                    viewers[k] = None
                    print(f"{now:<19} {hex(k):>8} {'?':>7} | "
                          f"{'NA':>6} {'NA':>6} {'NA':>6} {'NA':>8} {'NA':>8} || "
                          f"{'NA':>6} {'NA':>6} {'NA':>6} {'NA':>8} {'NA':>8}")
                    continue

                hstr = fmt_stats(hdr_mon, w_w=8, w_r=8)
                dstr = fmt_stats(data_mon, w_w=8, w_r=8)

                readers_str = str(n_readers) if n_readers is not None else "?"
                print(f"{now:<19} {hex(k):>8} {readers_str:>7} | {hstr} || {dstr}")

            time.sleep(max(args.interval, 0.01))

    except KeyboardInterrupt:
        pass
    finally:
        for v in viewers.values():
            try:
                if v is not None:
                    v.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
