# tools/parse_nsys.py
# Robust "student-friendly" parser for Nsight Systems CLI output.
#
# Usage:
#   python tools/parse_nsys.py results/nsys/stencil49.nsys-rep
#   python tools/parse_nsys.py results/nsys/stencil49.nsys-rep --top 5 --json
#
# Extracts essentials:
# - cuCtxSynchronize total time (CPU waiting on GPU)
# - total GPU kernel time + top kernels (from CUDA GPU Kernel Summary)
# - total memcpy time (HtoD + DtoH) (from CUDA GPU MemOps Summary by Time)
#
# The extractor is robust across Nsight Systems versions by searching
# for human-readable section headers (e.g. "CUDA API Summary").

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

# -------- parsing helpers --------

def _digits_only(s: str) -> int:
    """Parse integers like '2.024.046.783' or '295,173,472' safely (strip non-digits)."""
    if not s:
        return 0
    d = re.sub(r"[^\d]", "", s)
    return int(d) if d else 0

def _ns_to_ms(ns: int) -> float:
    return ns / 1e6

def _short_kernel_name(full: str) -> str:
    """
    Convert long Numba kernel names like:
      cudapy::__main__::stencil49_naive[abi:v1,...]
    into:
      stencil49_naive
    If no pattern matched, return a cleaned up last token.
    """
    if not full:
        return full
    # try to extract after last '::'
    if "::" in full:
        s = full.split("::")[-1]
    else:
        s = full
    # cut anything after '[' or whitespace separators used in nsight outputs
    s = re.split(r"[\[\s]", s)[0]
    return s.strip() if s.strip() else full

def _parse_table_rows_by_columns(section: str) -> List[List[str]]:
    """
    Turn a text table into rows of columns by splitting on 2+ spaces.
    Filters header/separator lines.
    """
    rows = []
    for line in section.splitlines():
        line = line.rstrip()
        if not line:
            continue
        # skip obvious header/separators
        if re.match(r"^\s*Time\s*\(%\)", line) or re.match(r"^-{3,}", line):
            continue
        # Heuristic: split by 2+ spaces to separate numeric columns from name
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) >= 2:
            rows.append(parts)
    return rows

@dataclass
class KernelRow:
    total_ns: int
    instances: int
    name: str

def _parse_gpukernsum(section: str) -> List[KernelRow]:
    rows: List[KernelRow] = []
    if not section or "SKIPPED" in section:
        return rows

    table_rows = _parse_table_rows_by_columns(section)
    for parts in table_rows:
        # Try to be tolerant: common formats have Total Time at index 1 and Instances at index 2
        # name is usually the last column
        try:
            if len(parts) >= 4:
                # e.g. [Time%, TotalTime, Instances, Avg, ... , Name]
                total_ns = _digits_only(parts[1])
                instances = _digits_only(parts[2])
                name = parts[-1]
            elif len(parts) == 3:
                total_ns = _digits_only(parts[1])
                instances = _digits_only(parts[2])
                name = parts[-1]
            else:
                continue
            if total_ns > 0 and name:
                rows.append(KernelRow(total_ns=total_ns, instances=instances, name=name))
        except Exception:
            continue

    # Sort by total time desc
    rows.sort(key=lambda r: r.total_ns, reverse=True)
    return rows

def _parse_cudaapisum_cuctxsynchronize(section: str) -> int:
    if not section or "SKIPPED" in section:
        return 0
    # scan lines for cuCtxSynchronize and parse the Total Time column
    for line in section.splitlines():
        if "cuCtxSynchronize" in line:
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) >= 2:
                return _digits_only(parts[1])
            # fallback: look for first large number in line
            m = re.search(r"(\d[\d\.,]+)", line)
            if m:
                return _digits_only(m.group(1))
    return 0

def _parse_gpumemtimesum(section: str) -> Dict[str, int]:
    """
    Returns dict with keys: "HtoD", "DtoH", "Total".
    Be tolerant to different labeling across NSYS versions.
    """
    out = {"HtoD": 0, "DtoH": 0, "Total": 0}
    if not section or "SKIPPED" in section:
        return out

    for line in section.splitlines():
        if "Host-to-Device" in line and "[CUDA memcpy" in line:
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) >= 2:
                out["HtoD"] += _digits_only(parts[1])
        if "Device-to-Host" in line and "[CUDA memcpy" in line:
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) >= 2:
                out["DtoH"] += _digits_only(parts[1])
        # older/newer versions may include tags like "[CUDA memcpy HtoD]" explicitly
        if "[CUDA memcpy HtoD]" in line:
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) >= 2:
                out["HtoD"] += _digits_only(parts[1])
        if "[CUDA memcpy DtoH]" in line:
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) >= 2:
                out["DtoH"] += _digits_only(parts[1])

    out["Total"] = out["HtoD"] + out["DtoH"]
    return out

# ---------------- robust section extraction ----------------

def _extract_section_by_keyword(text: str, keywords: List[str]) -> Optional[str]:
    """
    Find the first occurrence of any keyword in text (case-sensitive),
    then return the block from the line containing the keyword until the next
    top-level separator that Nsight uses ("Processing [") or end of text.
    """
    if not text:
        return None
    for kw in keywords:
        idx = text.find(kw)
        if idx != -1:
            # start at the line beginning containing the keyword
            start_line_idx = text.rfind("\n", 0, idx)
            start = start_line_idx + 1 if start_line_idx != -1 else idx
            # find next "Processing [" that typically precedes the next report block
            # search in the substring that follows the keyword start
            tail = text[idx:]
            m = re.search(r"\n\s*Processing \[", tail)
            end = idx + m.start() if m else len(text)
            return text[start:end].strip()
    return None

_report_key_map = {
    "cuda_api_sum": [
        "CUDA API Summary",
        "CUDA API Summary (cuda_api_sum)",
        "CUDA API Summary (cuda_api_sum):",
    ],
    "cuda_gpu_kern_sum": [
        "CUDA GPU Kernel Summary",
        "CUDA GPU Kernel Summary (cuda_gpu_kern_sum)",
        "CUDA GPU Kernel Summary (cuda_gpu_kern_time_sum)",
        "CUDA GPU Kernel Summary (cuda_gpu_kern_sum)",
    ],
    "cuda_gpu_mem_time_sum": [
        "CUDA GPU MemOps Summary (by Time)",
        "CUDA GPU MemOps Summary",
        "CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum)",
    ],
}

# -------- runner --------

def run_nsys_stats(path: str) -> str:
    if shutil.which("nsys") is None:
        raise RuntimeError("nsys not found in PATH. Load Nsight Systems or fix PATH.")
    # Use plain 'nsys stats' which accepts .nsys-rep (and .sqlite) and prints to stdout.
    cmd = ["nsys", "stats", "--force-export=true", path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        # still return stdout so we can attempt to parse what exists
        return p.stdout
    return p.stdout

# -------- main CLI --------

def main() -> int:
    ap = argparse.ArgumentParser(description="Student-friendly NSYS summary parser (CLI).")
    ap.add_argument("report", help="Path to .nsys-rep (recommended) or .sqlite generated by nsys stats")
    ap.add_argument("--top", type=int, default=3, help="How many top kernels to show")
    ap.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = ap.parse_args()

    try:
        text = run_nsys_stats(args.report)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Extract candidate sections using robust keywords map
    cudaapisum = _extract_section_by_keyword(text, _report_key_map["cuda_api_sum"]) or ""
    gpukernsum = _extract_section_by_keyword(text, _report_key_map["cuda_gpu_kern_sum"]) or ""
    gpumemtimesum = _extract_section_by_keyword(text, _report_key_map["cuda_gpu_mem_time_sum"]) or ""

    sync_ns = _parse_cudaapisum_cuctxsynchronize(cudaapisum)
    kernels = _parse_gpukernsum(gpukernsum)
    mem = _parse_gpumemtimesum(gpumemtimesum)

    total_kernel_ns = sum(k.total_ns for k in kernels)
    top_k = kernels[: max(0, args.top)]

    payload = {
        "cuCtxSynchronize_ns": sync_ns,
        "cuCtxSynchronize_ms": _ns_to_ms(sync_ns),
        "kernel_total_ns": total_kernel_ns,
        "kernel_total_ms": _ns_to_ms(total_kernel_ns),
        "memcpy_total_ns": mem["Total"],
        "memcpy_total_ms": _ns_to_ms(mem["Total"]),
        "memcpy_HtoD_ms": _ns_to_ms(mem["HtoD"]),
        "memcpy_DtoH_ms": _ns_to_ms(mem["DtoH"]),
        "top_kernels": [
            {
                "name": _short_kernel_name(k.name),
                "total_ms": _ns_to_ms(k.total_ns),
                "instances": k.instances,
            }
            for k in top_k
        ],
        "notes": [
            "Kernel times come from CUDA GPU Kernel Summary (gpukernsum).",
            "Memcpy times come from CUDA GPU MemOps Summary (by Time).",
            "cuCtxSynchronize is CPU-side waiting; high values often indicate synchronization points.",
        ],
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    # Text output (default)
    print("=== NSYS SUMMARY (student-friendly) ===")
    print(f"CPU wait (cuCtxSynchronize): {payload['cuCtxSynchronize_ms']:.3f} ms")
    print(f"GPU kernel total           : {payload['kernel_total_ms']:.3f} ms")
    print(f"GPU memcpy total           : {payload['memcpy_total_ms']:.3f} ms "
          f"(HtoD {payload['memcpy_HtoD_ms']:.3f} ms, DtoH {payload['memcpy_DtoH_ms']:.3f} ms)")
    print("")
    if top_k:
        print(f"Top {len(top_k)} kernels by total GPU time:")
        for i, k in enumerate(payload["top_kernels"], 1):
            avg_ms = (k["total_ms"] / k["instances"]) if k["instances"] else 0.0
            print(f"  {i}. {k['name']}")
            print(f"     total: {k['total_ms']:.3f} ms | instances: {k['instances']}")
            print(f"     avg per launch: {avg_ms:.3f} ms")
    else:
        print("No GPU kernels found in CUDA GPU Kernel Summary (did you trace CUDA? Use: nsys profile -t cuda,osrt ...)")
    print("======================================")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
