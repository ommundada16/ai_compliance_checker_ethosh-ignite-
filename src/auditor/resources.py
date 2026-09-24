"""Memory guard for long local runs.

Why this exists
---------------
On 2026-09-24 three evaluation jobs were left running at once on a 15.7 GB
laptop. Between them they loaded two ONNX models per process, an 8B model that
does not fit in 4 GB of VRAM and therefore spills into RAM, and a Docker VM.
Physical memory ran out, Windows began swapping, disk hit 100%, and the desktop
compositor was starved until the screen went black.

Nothing in the code noticed. Every process carried on requesting memory it was
never going to get, and the operating system absorbed the consequences.

A batch job that cannot finish should stop and say so, not take the machine
down with it. This turns "the laptop froze" into "the script exited with a
message", which is the difference between a bug and an outage.

Uses ctypes on Windows and /proc/meminfo on Linux rather than adding psutil:
one more dependency for a thirty-line check is not a good trade, and the guard
must never be the thing that fails to import.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


class OutOfMemoryGuard(Exception):
    """Raised when free memory falls below the floor. Stop, do not continue."""


@dataclass(frozen=True)
class Memory:
    total_gb: float
    available_gb: float

    @property
    def used_gb(self) -> float:
        return round(self.total_gb - self.available_gb, 2)

    @property
    def percent_used(self) -> float:
        return round(100.0 * self.used_gb / self.total_gb, 1) if self.total_gb else 0.0

    def __str__(self) -> str:
        return (
            f"{self.available_gb:.1f} GB free of {self.total_gb:.1f} GB "
            f"({self.percent_used:.0f}% used)"
        )


def read_memory() -> Memory | None:
    """Current physical memory, or None if it cannot be determined.

    Returning None rather than raising: a guard that crashes the run because it
    could not read a counter is worse than no guard at all.
    """
    try:
        if sys.platform == "win32":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            gb = 1024 ** 3
            return Memory(status.ullTotalPhys / gb, status.ullAvailPhys / gb)

        with open("/proc/meminfo", encoding="utf-8") as fh:
            values = {}
            for line in fh:
                key, _, rest = line.partition(":")
                values[key] = int(rest.strip().split()[0])  # kB
        mb = 1024 * 1024
        return Memory(
            values["MemTotal"] / mb,
            values.get("MemAvailable", values["MemFree"]) / mb,
        )
    except Exception:  # noqa: BLE001
        return None


def check_memory(floor_gb: float = 2.0, label: str = "") -> Memory | None:
    """Raise OutOfMemoryGuard if free memory is below `floor_gb`.

    The floor is headroom for the OPERATING SYSTEM, not for this process. Below
    roughly 2 GB Windows starts paging aggressively and the machine becomes
    unusable long before anything reports an error, so the check has to fire
    while there is still room to exit cleanly.
    """
    memory = read_memory()
    if memory is None:
        return None
    if memory.available_gb < floor_gb:
        where = f" before {label}" if label else ""
        raise OutOfMemoryGuard(
            f"stopping{where}: only {memory.available_gb:.1f} GB free, "
            f"floor is {floor_gb:.1f} GB. Close other applications, or run with "
            f"a cloud provider so the model does not load locally."
        )
    return memory


def describe_plan(steps: list[str], estimated_gb: float) -> str:
    """A one-screen summary printed before a long run starts.

    So the person at the keyboard can decide whether to let it run, rather than
    discovering what it loaded by watching Task Manager.
    """
    memory = read_memory()
    lines = ["", "=" * 66, "RESOURCE PLAN".center(66), "=" * 66]
    for step in steps:
        lines.append(f"  {step}")
    lines.append("-" * 66)
    lines.append(f"  estimated peak local RAM : ~{estimated_gb:.1f} GB")
    if memory:
        lines.append(f"  available right now      : {memory}")
        headroom = memory.available_gb - estimated_gb
        verdict = "OK" if headroom > 2.0 else "TIGHT -- consider a cloud provider"
        lines.append(f"  headroom                 : {headroom:.1f} GB  [{verdict}]")
    lines.append("=" * 66)
    return "\n".join(lines)
