#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cbf_live_monitor_v1.0.py

Real-time / near-real-time monitoring for the 4-antenna PSRDADA + CUDA FXB(+V) pipeline.

Data sources:
  1) tracking CSV produced by bf_fxbv_stream_dumpvis_cuda (v1.5.x):
       --track-csv <file>
     Columns (expected):
       utc_iso,t_data_sec,ant_id,az_deg,el_deg,tau_ns,coarse_samp,fine_ns,
       coh_xx,snr_xx,snr_xx_rad,snr_xx_emp,p_ref_xx,p_ant_xx,tau_res_xx_ns,
       coh_yy,snr_yy,snr_yy_rad,snr_yy_emp,p_ref_yy,p_ant_yy,tau_res_yy_ns

  2) Optional VISDUMP binary (for spectrum phase/coherence) produced by:
       --dump-visdir <dir>  --dump-ref 0  --dump-pols XX,YY
     Files like: <visdir>/vis_0-2_XX.bin  etc.

Features:
  - Time series: coherence, SNR, residual delay, power, az/el (per ant)
  - Optional spectrum panel (latest VISDUMP record): coherence(f) + phase(f)
  - Headless mode: periodically saves PNG; optional HTML auto-refresh page

Typical usage (GUI if DISPLAY exists):
  python3 cbf_live_monitor_v1.0.py --track-csv viscal_256/track.csv --visdir viscal_256 --baseline 0-2 --pol XX --fcenter-hz 1.42e9

Headless (save PNG and an auto-refresh HTML):
  python3 cbf_live_monitor_v1.0.py --track-csv viscal_256/track.csv --visdir viscal_256 --baseline 0-2 --pol XX \
    --fcenter-hz 1.42e9 --out status.png --html status.html --update-sec 1

Then serve in another terminal:
  python3 -m http.server 8000
Open: http://<host>:8000/status.html

Notes:
  - This script is intentionally "best-effort" and tolerant to missing columns / empty files.
  - VISDUMP reading always reads the last COMPLETE record (ignores partial trailing record).
"""

import argparse
import csv
import os
import sys
import time
import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Decide backend early (before importing pyplot)
NO_GUI = ("--no-gui" in sys.argv) or (os.environ.get("DISPLAY", "") == "")
if NO_GUI:
    import matplotlib
    matplotlib.use("Agg")  # headless

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone


# ------------------------- helpers -------------------------

def _parse_iso_utc(s: str) -> Optional[datetime]:
    """Parse utc_iso from track.csv. Returns timezone-aware datetime in UTC if possible."""
    s = (s or "").strip()
    if not s:
        return None
    # common: "2026-02-19T21:21:07.123Z" or without Z
    try:
        if s.endswith("Z"):
            s2 = s[:-1]
            dt = datetime.fromisoformat(s2)
            return dt.replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(s)
        # if naive, assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _f(x: str) -> float:
    try:
        if x is None:
            return float("nan")
        x = x.strip()
        if x == "" or x.lower() == "nan":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _i(x: str) -> int:
    try:
        return int(float(x))
    except Exception:
        return -999999


def _pol_norm(pol: str) -> str:
    pol = (pol or "").strip().upper()
    if pol in ("XX", "YY", "XY", "YX"):
        return pol
    return "XX"


def _baseline_parse(s: str) -> Tuple[int, int]:
    s = (s or "").strip()
    for sep in ("-", ",", ":"):
        if sep in s:
            a, b = s.split(sep, 1)
            return int(a), int(b)
    raise ValueError("bad baseline, expect like 0-2")


# ------------------------- VISDUMP reader -------------------------

@dataclass
class VisdumpHeader:
    nfft: int
    fs_hz: float
    monitor_sec: float
    bi: int
    bj: int
    pol_code: int  # 0=XX 1=YY 2=XY 3=YX

def _read_visdump_header(path: str) -> VisdumpHeader:
    with open(path, "rb") as f:
        hdr = f.read(64)
    if len(hdr) < 64:
        raise RuntimeError(f"{path}: short header {len(hdr)} bytes")
    magic = hdr[:8]
    if magic != b"VISDUMP1":
        raise RuntimeError(f"{path}: bad magic {magic!r}")
    # struct layout matches bf_fxbv_stream_dumpvis_cuda_v1.5.x:
    # 8s I d d I I I I pad...
    nfft = struct.unpack_from("<I", hdr, 8)[0]
    fs_hz = struct.unpack_from("<d", hdr, 12)[0]
    monitor_sec = struct.unpack_from("<d", hdr, 20)[0]
    bi = struct.unpack_from("<I", hdr, 28)[0]
    bj = struct.unpack_from("<I", hdr, 32)[0]
    pol_code = struct.unpack_from("<I", hdr, 36)[0]
    return VisdumpHeader(nfft=nfft, fs_hz=fs_hz, monitor_sec=monitor_sec, bi=bi, bj=bj, pol_code=pol_code)

def _visdump_record_size(nfft: int) -> int:
    # double t + float coh + float pad + cf32[nfft] + double[nfft] + double[nfft]
    return (8 + 4 + 4) + (8 * nfft) + (8 * nfft) + (8 * nfft)

def _read_last_visdump_record(path: str, nfft: int) -> Optional[Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns (t_mono, coh_avg, vis_cf32[nfft] complex64, p0[nfft] float64, p1[nfft] float64)
    or None if no complete record yet.
    """
    rec_size = _visdump_record_size(nfft)
    st = os.stat(path)
    if st.st_size < 64 + rec_size:
        return None
    nrec = (st.st_size - 64) // rec_size
    if nrec <= 0:
        return None
    off = 64 + (nrec - 1) * rec_size
    with open(path, "rb") as f:
        f.seek(off)
        buf = f.read(rec_size)
    if len(buf) < rec_size:
        return None
    t_mono = struct.unpack_from("<d", buf, 0)[0]
    coh_avg = struct.unpack_from("<f", buf, 8)[0]
    # skip pad float at 12
    # vis cf32 = {float re, float im} * nfft
    vis = np.frombuffer(buf, dtype=np.float32, count=2 * nfft, offset=16).reshape(nfft, 2)
    vis_c = vis[:, 0].astype(np.float32) + 1j * vis[:, 1].astype(np.float32)
    # p0 and p1 follow
    p0_off = 16 + 8 * nfft
    p0 = np.frombuffer(buf, dtype=np.float64, count=nfft, offset=p0_off).copy()
    p1_off = p0_off + 8 * nfft
    p1 = np.frombuffer(buf, dtype=np.float64, count=nfft, offset=p1_off).copy()
    return t_mono, float(coh_avg), vis_c.astype(np.complex64), p0, p1


# ------------------------- track.csv tailer -------------------------

@dataclass
class TrackRow:
    utc: Optional[datetime]
    t_data_sec: float
    ant_id: int
    az_deg: float
    el_deg: float
    tau_ns: float
    coarse_samp: int
    fine_ns: float
    coh_xx: float
    snr_xx: float
    snr_xx_rad: float
    snr_xx_emp: float
    p_ref_xx: float
    p_ant_xx: float
    tau_res_xx_ns: float
    coh_yy: float
    snr_yy: float
    snr_yy_rad: float
    snr_yy_emp: float
    p_ref_yy: float
    p_ant_yy: float
    tau_res_yy_ns: float

def _parse_track_row(cols: List[str], values: List[str]) -> TrackRow:
    d = {c: (values[i] if i < len(values) else "") for i, c in enumerate(cols)}
    utc = _parse_iso_utc(d.get("utc_iso", ""))
    return TrackRow(
        utc=utc,
        t_data_sec=_f(d.get("t_data_sec", "nan")),
        ant_id=_i(d.get("ant_id", "-1")),
        az_deg=_f(d.get("az_deg", "nan")),
        el_deg=_f(d.get("el_deg", "nan")),
        tau_ns=_f(d.get("tau_ns", "nan")),
        coarse_samp=_i(d.get("coarse_samp", "0")),
        fine_ns=_f(d.get("fine_ns", "nan")),
        coh_xx=_f(d.get("coh_xx", "nan")),
        snr_xx=_f(d.get("snr_xx", "nan")),
        snr_xx_rad=_f(d.get("snr_xx_rad", "nan")),
        snr_xx_emp=_f(d.get("snr_xx_emp", "nan")),
        p_ref_xx=_f(d.get("p_ref_xx", "nan")),
        p_ant_xx=_f(d.get("p_ant_xx", "nan")),
        tau_res_xx_ns=_f(d.get("tau_res_xx_ns", "nan")),
        coh_yy=_f(d.get("coh_yy", "nan")),
        snr_yy=_f(d.get("snr_yy", "nan")),
        snr_yy_rad=_f(d.get("snr_yy_rad", "nan")),
        snr_yy_emp=_f(d.get("snr_yy_emp", "nan")),
        p_ref_yy=_f(d.get("p_ref_yy", "nan")),
        p_ant_yy=_f(d.get("p_ant_yy", "nan")),
        tau_res_yy_ns=_f(d.get("tau_res_yy_ns", "nan")),
    )

@dataclass
class TrackTailer:
    path: str
    cols: List[str] = field(default_factory=list)
    fp: Optional[object] = None
    pos: int = 0

    def open_and_init(self, from_start: bool) -> None:
        # Wait until file exists and has at least header
        while True:
            if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
                break
            time.sleep(0.5)

        self.fp = open(self.path, "r", newline="")
        header = self.fp.readline()
        if not header:
            raise RuntimeError(f"{self.path}: empty")
        self.cols = [c.strip() for c in header.strip().split(",")]
        if not from_start:
            # seek to end (tail)
            self.fp.seek(0, os.SEEK_END)
        self.pos = self.fp.tell()

    def read_new_rows(self) -> List[TrackRow]:
        if self.fp is None:
            return []
        # handle truncation / rotation
        try:
            st = os.stat(self.path)
            if st.st_size < self.pos:
                # file was truncated; reopen
                try:
                    self.fp.close()
                except Exception:
                    pass
                self.open_and_init(from_start=False)
        except Exception:
            pass

        self.fp.seek(self.pos)
        lines = self.fp.readlines()
        self.pos = self.fp.tell()

        rows: List[TrackRow] = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            # ignore malformed
            vals = [x.strip() for x in ln.split(",")]
            try:
                row = _parse_track_row(self.cols, vals)
                rows.append(row)
            except Exception:
                continue
        return rows


# ------------------------- dashboard logic -------------------------

@dataclass
class Series:
    t: List[datetime] = field(default_factory=list)
    coh_xx: List[float] = field(default_factory=list)
    coh_yy: List[float] = field(default_factory=list)
    snr_xx: List[float] = field(default_factory=list)
    snr_yy: List[float] = field(default_factory=list)
    tau_res_xx_ns: List[float] = field(default_factory=list)
    tau_res_yy_ns: List[float] = field(default_factory=list)
    p_ref_xx: List[float] = field(default_factory=list)
    p_ant_xx: List[float] = field(default_factory=list)
    p_ref_yy: List[float] = field(default_factory=list)
    p_ant_yy: List[float] = field(default_factory=list)
    az_deg: List[float] = field(default_factory=list)
    el_deg: List[float] = field(default_factory=list)
    tau_ns: List[float] = field(default_factory=list)
    coarse_samp: List[int] = field(default_factory=list)
    fine_ns: List[float] = field(default_factory=list)

def _trim_by_time(s: Series, window_sec: float) -> None:
    if not s.t:
        return
    tmax = s.t[-1]
    tmin = tmax.timestamp() - window_sec
    # find first idx >= tmin
    idx = 0
    for i, tt in enumerate(s.t):
        if tt.timestamp() >= tmin:
            idx = i
            break
    if idx <= 0:
        return
    # trim all lists
    for name in s.__dataclass_fields__.keys():
        lst = getattr(s, name)
        if isinstance(lst, list) and len(lst) >= idx:
            setattr(s, name, lst[idx:])

def _guess_snr_mode(last_row: TrackRow) -> str:
    # compare snr_xx to rad/emp if available
    if (not math.isnan(last_row.snr_xx)) and (not math.isnan(last_row.snr_xx_rad)) and (not math.isnan(last_row.snr_xx_emp)):
        if abs(last_row.snr_xx - last_row.snr_xx_rad) < abs(last_row.snr_xx - last_row.snr_xx_emp):
            return "rad"
        else:
            return "emp"
    return "unknown"


def _safe_db(x: float) -> float:
    if not np.isfinite(x) or x <= 0:
        return float("nan")
    return 10.0 * math.log10(x)


def write_autorefresh_html(html_path: str, png_path: str, title: str = "CBF Monitor", interval_ms: int = 1000) -> None:
    # Simple standalone HTML that reloads the PNG by cache-busting query string.
    png_base = os.path.basename(png_path)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: sans-serif; background: #111; color: #ddd; }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 12px; }}
    img {{ width: 100%; height: auto; border: 1px solid #333; }}
    .small {{ font-size: 12px; opacity: 0.8; }}
  </style>
</head>
<body>
<div class="wrap">
  <h2>{title}</h2>
  <div class="small">Auto-refresh every {interval_ms} ms. File: {png_base}</div>
  <img id="img" src="{png_base}?t=0">
</div>
<script>
  const img = document.getElementById("img");
  function refresh() {{
    img.src = "{png_base}?t=" + Date.now();
  }}
  setInterval(refresh, {interval_ms});
  refresh();
</script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser(description="Real-time monitoring plots for bf_fxbv_stream_dumpvis_cuda track.csv (+ optional visdump)")
    ap.add_argument("--track-csv", required=True, help="Tracking CSV produced by bf (--track-csv)")
    ap.add_argument("--visdir", default="", help="VISDUMP directory (from --dump-visdir). If set, spectrum panel enabled.")
    ap.add_argument("--vis", default="", help="Direct VISDUMP file path (overrides --visdir/--baseline/--pol)")
    ap.add_argument("--baseline", default="0-1", help="Baseline for spectrum panel, like 0-2 (default 0-1)")
    ap.add_argument("--pol", default="XX", help="Pol for spectrum panel: XX/YY/XY/YX (default XX)")
    ap.add_argument("--fcenter-hz", type=float, default=float("nan"), help="RF center frequency for absolute frequency axis (Hz)")
    ap.add_argument("--freq-axis", choices=["abs", "bb"], default="abs", help="Spectrum x-axis: abs (Hz) or bb (baseband Hz)")
    ap.add_argument("--fftshift", type=int, default=1, help="1 to fftshift spectrum plot (default 1)")
    ap.add_argument("--update-sec", type=float, default=1.0, help="Update interval seconds (default 1.0)")
    ap.add_argument("--window-sec", type=float, default=300.0, help="History window seconds for time series (default 300)")
    ap.add_argument("--ants", default="1,2,3", help="Which ant_id to plot vs ref=0, comma list (default 1,2,3)")
    ap.add_argument("--out", default="", help="If set, save PNG to this path each update (headless-friendly)")
    ap.add_argument("--html", default="", help="If set, write an auto-refresh HTML that shows --out PNG")
    ap.add_argument("--no-gui", type=int, default=0, help="Force headless mode (Agg); useful without DISPLAY (default 0)")
    ap.add_argument("--from-start", type=int, default=0, help="1 to read track.csv from start instead of tail (default 0)")
    args = ap.parse_args()

    ants = [int(x) for x in args.ants.split(",") if x.strip() != ""]
    ants = [a for a in ants if a >= 0]

    # Determine vis path if needed
    vis_path = ""
    if args.vis:
        vis_path = args.vis
    elif args.visdir:
        bi, bj = _baseline_parse(args.baseline)
        pol = _pol_norm(args.pol)
        vis_path = os.path.join(args.visdir, f"vis_{bi}-{bj}_{pol}.bin")

    # Tailer
    tailer = TrackTailer(args.track_csv)
    tailer.open_and_init(from_start=bool(args.from_start))

    # Data store per ant_id
    series: Dict[int, Series] = {a: Series() for a in ants}
    last_row_any: Optional[TrackRow] = None

    # Setup plot
    fig = plt.figure(figsize=(14, 9), dpi=110)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.2])

    ax_coh = fig.add_subplot(gs[0, 0])
    ax_snr = fig.add_subplot(gs[0, 1])
    ax_tau = fig.add_subplot(gs[1, 0])
    ax_pwr = fig.add_subplot(gs[1, 1])
    ax_azel = fig.add_subplot(gs[2, 0])
    ax_spec = fig.add_subplot(gs[2, 1])

    fig.suptitle("CBF Live Monitor (track.csv + optional VISDUMP)", fontsize=14)

    # Pre-create line objects for speed
    lines = {}

    def make_lines():
        for a in ants:
            # coherence
            (l1,) = ax_coh.plot([], [], label=f"cohXX 0-{a}")
            (l2,) = ax_coh.plot([], [], label=f"cohYY 0-{a}", linestyle="--")
            # snr
            (l3,) = ax_snr.plot([], [], label=f"snrXX 0-{a}")
            (l4,) = ax_snr.plot([], [], label=f"snrYY 0-{a}", linestyle="--")
            # residual delay
            (l5,) = ax_tau.plot([], [], label=f"tauResXX(ns) 0-{a}")
            (l6,) = ax_tau.plot([], [], label=f"tauResYY(ns) 0-{a}", linestyle="--")
            # power (dB)
            (l7,) = ax_pwr.plot([], [], label=f"PrefXX(dB) a0 (from 0-{a})")
            (l8,) = ax_pwr.plot([], [], label=f"PantXX(dB) a{a}")
            (l9,) = ax_pwr.plot([], [], label=f"PrefYY(dB) a0", linestyle="--")
            (l10,) = ax_pwr.plot([], [], label=f"PantYY(dB) a{a}", linestyle="--")
            lines[(a, "coh_xx")] = l1
            lines[(a, "coh_yy")] = l2
            lines[(a, "snr_xx")] = l3
            lines[(a, "snr_yy")] = l4
            lines[(a, "tau_res_xx_ns")] = l5
            lines[(a, "tau_res_yy_ns")] = l6
            lines[(a, "p_ref_xx_db")] = l7
            lines[(a, "p_ant_xx_db")] = l8
            lines[(a, "p_ref_yy_db")] = l9
            lines[(a, "p_ant_yy_db")] = l10

        (lspec1,) = ax_spec.plot([], [], label="|V|/sqrt(P0P1) (coh_spec)")
        ax_spec2 = ax_spec.twinx()
        (lspec2,) = ax_spec2.plot([], [], label="phase(V) rad", linestyle="--")
        lines[("spec", "coh")] = lspec1
        lines[("spec", "phase")] = lspec2
        lines[("spec", "ax2")] = ax_spec2

    make_lines()

    for ax in (ax_coh, ax_snr, ax_tau, ax_pwr, ax_azel, ax_spec):
        ax.grid(True, alpha=0.25)

    ax_coh.set_ylabel("coherence")
    ax_coh.set_ylim(0, 1.05)
    ax_coh.legend(loc="upper right", fontsize=8, ncol=2)

    ax_snr.set_ylabel("SNR")
    ax_snr.legend(loc="upper right", fontsize=8, ncol=2)

    ax_tau.set_ylabel("residual delay (ns)")
    ax_tau.legend(loc="upper right", fontsize=8, ncol=2)

    ax_pwr.set_ylabel("power (dB)")
    ax_pwr.legend(loc="upper right", fontsize=8, ncol=2)

    ax_azel.set_ylabel("deg")
    ax_azel.set_xlabel("UTC")
    ax_azel.set_title("Az/El (per ant)")
    az_lines = {}
    el_lines = {}
    for a in ants:
        (la,) = ax_azel.plot([], [], label=f"az a{a}")
        (le,) = ax_azel.plot([], [], label=f"el a{a}", linestyle="--")
        az_lines[a] = la
        el_lines[a] = le
    ax_azel.legend(loc="upper right", fontsize=8, ncol=2)

    ax_spec.set_title("Spectrum (latest VISDUMP record)")
    ax_spec.set_xlabel("frequency (Hz)" if args.freq_axis == "abs" else "baseband (Hz)")
    ax_spec.set_ylabel("coherence spectrum")
    lines[("spec", "ax2")].set_ylabel("phase (rad)")

    # Date formatting for x-axes
    for ax in (ax_coh, ax_snr, ax_tau, ax_pwr, ax_azel):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.tick_params(axis="x", rotation=0)

    meta_text = fig.text(0.01, 0.01, "", fontsize=9, family="monospace", va="bottom")

    # Optionally write HTML for auto-refresh
    if args.html and args.out:
        write_autorefresh_html(args.html, args.out, title="CBF Monitor", interval_ms=int(args.update_sec * 1000))

    # Main loop
    last_vis_hdr: Optional[VisdumpHeader] = None

    if (not NO_GUI) and (not args.no_gui) and (not args.out):
        plt.ion()
        plt.show()

    while True:
        # ingest new track rows
        new_rows = tailer.read_new_rows()
        if new_rows:
            for r in new_rows:
                last_row_any = r
                if r.ant_id in series and r.utc is not None:
                    s = series[r.ant_id]
                    s.t.append(r.utc)
                    s.coh_xx.append(r.coh_xx)
                    s.coh_yy.append(r.coh_yy)
                    s.snr_xx.append(r.snr_xx)
                    s.snr_yy.append(r.snr_yy)
                    s.tau_res_xx_ns.append(r.tau_res_xx_ns)
                    s.tau_res_yy_ns.append(r.tau_res_yy_ns)
                    s.p_ref_xx.append(r.p_ref_xx)
                    s.p_ant_xx.append(r.p_ant_xx)
                    s.p_ref_yy.append(r.p_ref_yy)
                    s.p_ant_yy.append(r.p_ant_yy)
                    s.az_deg.append(r.az_deg)
                    s.el_deg.append(r.el_deg)
                    s.tau_ns.append(r.tau_ns)
                    s.coarse_samp.append(r.coarse_samp)
                    s.fine_ns.append(r.fine_ns)
                    _trim_by_time(s, args.window_sec)

        # Update time-series plots
        for a in ants:
            s = series[a]
            if not s.t:
                continue
            t = mdates.date2num(s.t)

            # coherence
            lines[(a, "coh_xx")].set_data(t, np.array(s.coh_xx, dtype=float))
            lines[(a, "coh_yy")].set_data(t, np.array(s.coh_yy, dtype=float))

            # snr
            lines[(a, "snr_xx")].set_data(t, np.array(s.snr_xx, dtype=float))
            lines[(a, "snr_yy")].set_data(t, np.array(s.snr_yy, dtype=float))

            # residual delay
            lines[(a, "tau_res_xx_ns")].set_data(t, np.array(s.tau_res_xx_ns, dtype=float))
            lines[(a, "tau_res_yy_ns")].set_data(t, np.array(s.tau_res_yy_ns, dtype=float))

            # power (dB)
            pref_xx_db = np.array([_safe_db(x) for x in s.p_ref_xx], dtype=float)
            pant_xx_db = np.array([_safe_db(x) for x in s.p_ant_xx], dtype=float)
            pref_yy_db = np.array([_safe_db(x) for x in s.p_ref_yy], dtype=float)
            pant_yy_db = np.array([_safe_db(x) for x in s.p_ant_yy], dtype=float)
            lines[(a, "p_ref_xx_db")].set_data(t, pref_xx_db)
            lines[(a, "p_ant_xx_db")].set_data(t, pant_xx_db)
            lines[(a, "p_ref_yy_db")].set_data(t, pref_yy_db)
            lines[(a, "p_ant_yy_db")].set_data(t, pant_yy_db)

            # az/el
            az_lines[a].set_data(t, np.array(s.az_deg, dtype=float))
            el_lines[a].set_data(t, np.array(s.el_deg, dtype=float))

        # Autoscale x-lims based on available data (use latest timestamp among series)
        all_times = []
        for a in ants:
            if series[a].t:
                all_times.append(series[a].t[-1])
        if all_times:
            tmax = max(all_times)
            tmin = datetime.fromtimestamp(tmax.timestamp() - args.window_sec, tz=timezone.utc)
            xmin = mdates.date2num(tmin)
            xmax = mdates.date2num(tmax)

            for ax in (ax_coh, ax_snr, ax_tau, ax_pwr, ax_azel):
                ax.set_xlim(xmin, xmax)

        # y autoscale for SNR / tau / pwr
        ax_snr.relim(); ax_snr.autoscale_view(scalex=False, scaley=True)
        ax_tau.relim(); ax_tau.autoscale_view(scalex=False, scaley=True)
        ax_pwr.relim(); ax_pwr.autoscale_view(scalex=False, scaley=True)

        # Spectrum panel
        spec_msg = ""
        if vis_path:
            try:
                if last_vis_hdr is None:
                    last_vis_hdr = _read_visdump_header(vis_path)
                hdr = last_vis_hdr
                rec = _read_last_visdump_record(vis_path, hdr.nfft)
                if rec is not None:
                    t_mono, coh_avg, vis_c, p0, p1 = rec

                    # freq axis
                    fs = hdr.fs_hz
                    nfft = hdr.nfft
                    fbin = np.fft.fftfreq(nfft, d=1.0/fs)  # baseband Hz
                    if args.fftshift:
                        fbin = np.fft.fftshift(fbin)
                        vis_c = np.fft.fftshift(vis_c)
                        p0 = np.fft.fftshift(p0)
                        p1 = np.fft.fftshift(p1)

                    if args.freq_axis == "abs":
                        f0 = args.fcenter_hz
                        if not np.isfinite(f0):
                            f0 = 0.0
                        fx = f0 + fbin
                    else:
                        fx = fbin

                    coh_spec = np.abs(vis_c) / np.sqrt(np.maximum(p0*p1, 1e-30))
                    phase = np.angle(vis_c)

                    lines[("spec", "coh")].set_data(fx, coh_spec)
                    lines[("spec", "phase")].set_data(fx, phase)

                    # scale axes
                    ax_spec.set_xlim(float(np.nanmin(fx)), float(np.nanmax(fx)))
                    ax_spec.set_ylim(0, 1.05)
                    lines[("spec", "ax2")].set_ylim(-math.pi, math.pi)

                    spec_msg = f"VIS {os.path.basename(vis_path)}  nfft={hdr.nfft}  fs={hdr.fs_hz/1e6:.3f}MHz  df={hdr.fs_hz/hdr.nfft/1e3:.3f}kHz  coh_avg={coh_avg:.3f}"
                else:
                    spec_msg = f"VIS {os.path.basename(vis_path)}: waiting for records..."
            except Exception as e:
                spec_msg = f"VIS: {e}"

        # Meta / status text
        if last_row_any and last_row_any.utc:
            snr_mode = _guess_snr_mode(last_row_any)
            meta = [
                f"UTC(now)={last_row_any.utc.strftime('%Y-%m-%d %H:%M:%S')}Z   snr_mode≈{snr_mode}",
                f"track_csv={args.track_csv}",
            ]
            if np.isfinite(args.fcenter_hz):
                meta.append(f"fcenter={args.fcenter_hz/1e6:.3f} MHz")
            if last_vis_hdr:
                meta.append(f"visdump: nfft={last_vis_hdr.nfft} fs={last_vis_hdr.fs_hz/1e6:.3f}MHz df={last_vis_hdr.fs_hz/last_vis_hdr.nfft/1e3:.3f}kHz mon={last_vis_hdr.monitor_sec:.3f}s")
            if spec_msg:
                meta.append(spec_msg)
            meta_text.set_text("\n".join(meta))

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Render
        if args.out:
            fig.savefig(args.out, dpi=110)
        if (not NO_GUI) and (not args.no_gui) and (not args.out):
            fig.canvas.draw()
            plt.pause(0.001)

        time.sleep(max(args.update_sec, 0.05))


if __name__ == "__main__":
    main()
