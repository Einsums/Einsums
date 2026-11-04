# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""``einsums info``: what this Einsums is and what it runs on, for a bug report.

Reads the build configuration compiled into ``_core`` without starting the runtime. The hardware
profile is the one fact that needs the runtime, so it is read by a child process, and a runtime
that will not start is reported rather than taking the report down with it.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

#: Optional packages the tools use, and what each one adds.
OPTIONAL = {
    "numpy": "array interop",
    "textual": "einsums profiler",
    "textual_plotext": "profiler timeline and roofline plots",
    "zeroconf": "profiler server discovery over mDNS",
}

# x86-64 micro-architecture levels by the /proc/cpuinfo flags each requires (psABI).
_X86_LEVELS = (
    ("v4", {"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"}),
    ("v3", {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "movbe", "abm"}),
    ("v2", {"cx16", "lahf_lm", "popcnt", "sse4_1", "sse4_2", "ssse3"}),
)


def register(subparsers) -> None:
    p = subparsers.add_parser("info", help="Show the build, the Python environment, and the machine")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.set_defaults(func=run)


def _cpu() -> dict[str, object]:
    info: dict[str, object] = {"machine": platform.machine(), "logical_cores": os.cpu_count()}
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        cpuinfo = ""
    if model := re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M):
        info["model"] = model.group(1).strip()
    if flags := re.search(r"^flags\s*:\s*(.+)$", cpuinfo, re.M):
        have = set(flags.group(1).split())
        info["x86_64_level"] = next((level for level, need in _X86_LEVELS if need <= have), "baseline")
    elif sys.platform == "darwin":
        try:
            info["model"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    info["simd_override"] = {k: os.environ[k] for k in ("EINSUMS_SIMD_ARCH", "STRIPES_ARCH") if k in os.environ}
    return info


def _rpaths(binary: Path) -> list[str]:
    """The LC_RPATH entries of a Mach-O *binary*, with @loader_path expanded."""
    out = subprocess.run(["otool", "-l", str(binary)], capture_output=True, text=True).stdout
    paths = re.findall(r"cmd LC_RPATH\n\s*cmdsize \d+\n\s*path (.+?) \(offset \d+\)", out)
    return [p.replace("@loader_path", str(binary.parent)) for p in paths]


def link_walk_tool() -> str | None:
    """The tool that lists a binary's linked libraries here: otool on macOS, ldd on other Unixes.

    None on Windows, which has no such tool of its own. The ldd that Git for Windows puts on PATH is
    MSYS's, and prints MSYS paths that Windows Python cannot open, so it is not used even when present.
    """
    if sys.platform == "win32":
        return None
    tool = "otool" if sys.platform == "darwin" else "ldd"
    return tool if shutil.which(tool) else None


def _linked_libraries(binary: Path, inherited: tuple[str, ...] = ()) -> list[tuple[str, tuple[str, ...]]]:
    """Shared libraries *binary* resolves to, from ``link_walk_tool()`` (empty without one).

    Each comes with the rpaths a library it loads in turn searches. otool prints install names as the
    binary records them, so an @rpath or @loader_path one is expanded here the way dyld would, against
    *binary*'s own rpaths and those of whatever loaded it, rather than against the working directory.
    """
    tool = link_walk_tool()
    if tool == "otool":
        out = subprocess.run(["otool", "-L", str(binary)], capture_output=True, text=True).stdout
        search = (*_rpaths(binary), *inherited)
        found = []
        for name in (line.split()[0] for line in out.splitlines()[1:] if line.strip()):
            if name.startswith("@rpath/"):
                candidates = [Path(p) / name.removeprefix("@rpath/") for p in search]
            else:
                candidates = [Path(name.replace("@loader_path", str(binary.parent)))]
            hit = next((c for c in candidates if c.exists()), None)
            # Unresolved, so a stand-in such as conda's libblas.3.dylib keeps its own name.
            found.append((str(hit) if hit else name, search))
        return found
    if tool == "ldd":
        out = subprocess.run(["ldd", str(binary)], capture_output=True, text=True).stdout
        return [(m.group(1), ()) for m in re.finditer(r"=>\s+(\S+)", out)]
    return []


def _blas_libraries(libraries: list[str]) -> list[str]:
    pattern = re.compile(r"(openblas|mkl|blis|flexiblas|accelerate|essl|armpl|libblas|libcblas|liblapack)", re.I)
    return sorted({Path(lib).name for lib in libraries if pattern.search(Path(lib).name)})


# The BLAS is named by the runtime, from the library that actually supplies dgemm in this process: the
# same name a hardware profile records and checks. Linked libraries say only what a binary asked for,
# and conda's libblas, for one, is a stand-in for whichever implementation the environment selected.
_PROFILE_PROBE = (
    "import json; from einsums import hardware as hw; "
    "print(json.dumps({'state': hw.profile_state(), 'path': hw.profile_path(), "
    "'reason': hw.profile_reason(), 'fingerprint': hw.machine_fingerprint(), "
    "'blas': json.loads(hw.profile_header_json())['software']['blas']}))"
)


def _profile() -> dict[str, object]:
    """The hardware profile in effect, as a fresh process with this environment sees it."""
    from .. import __file__ as package_init

    # The child imports this same package, which the `einsums` launcher put on sys.path rather
    # than in the environment.
    search = os.pathsep.join(filter(None, [str(Path(package_init).resolve().parent.parent), os.environ.get("PYTHONPATH")]))
    try:
        out = subprocess.run([sys.executable, "-c", _PROFILE_PROBE], capture_output=True, text=True, timeout=120,
                             env={**os.environ, "PYTHONPATH": search})
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"state": "unknown", "reason": f"could not start the runtime: {error}"}
    if out.returncode != 0:
        tail = out.stderr.strip().splitlines()[-1:] or ["no message"]
        return {"state": "unknown", "reason": f"the runtime failed to start: {tail[0]}"}
    return json.loads(out.stdout.strip().splitlines()[-1])


def collect() -> dict[str, object]:
    from .. import _core

    version = _core._version
    config = {
        m.group(1): m.group(2) == "ON"
        for m in re.finditer(r"^\s*(EINSUMS_WITH_\w+)=(\w+)", version.configuration_string(), re.M)
    }
    full = version.full_build_string()
    git = re.search(r"Git:\s*(\w+)", full)
    core_path = Path(_core.__file__).resolve()
    linked = _linked_libraries(core_path)
    einsums_lib, search = next(((lib, s) for lib, s in linked if "libEinsums" in Path(lib).name), (None, ()))
    einsums_lib = str(Path(einsums_lib).resolve()) if einsums_lib and Path(einsums_lib).is_file() else None
    libraries = [lib for lib, _ in linked]
    if einsums_lib:  # BLAS hangs off libEinsums, not off _core
        libraries += [lib for lib, _ in _linked_libraries(Path(einsums_lib), search)]
    profile = _profile()
    blas = profile.pop("blas", None)
    optional = {}
    for module, purpose in OPTIONAL.items():
        if importlib.util.find_spec(module) is None:
            optional[module] = {"version": None, "for": purpose}
            continue
        try:
            found = importlib.metadata.version(module.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            found = "installed"
        optional[module] = {"version": found, "for": purpose}
    return {
        "einsums": {
            "version": version.full_version_as_string(),
            "git_commit": git.group(1) if git else None,
            "build_type": version.build_type(),
            "build_date": version.build_date_time(),
            "package": str(core_path.parent),
            "libEinsums": einsums_lib,
            "blas": blas if blas != "unidentified" else None,
            "blas_libraries": _blas_libraries(libraries),
            "options": config,
        },
        "python": {"version": platform.python_version(), "executable": sys.executable, "optional": optional},
        "platform": {"system": platform.platform()},
        "cpu": _cpu(),
        "profile": profile,
    }


def _blas_row(e: dict[str, object]) -> str:
    libraries = ", ".join(e["blas_libraries"])
    if e["blas"]:
        # Name the linked libraries only when they are stand-ins the runtime name does not already show.
        shown = all(lib in e["blas"] for lib in e["blas_libraries"])
        return e["blas"] if shown or not libraries else f"{e['blas']}, linked through {libraries}"
    return f"linked {libraries}; the runtime could not name the implementation" if libraries else "(could not tell)"


def run(args: argparse.Namespace) -> int:
    info = collect()
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    e, py, cpu = info["einsums"], info["python"], info["cpu"]
    rows = [
        ("Einsums", f"{e['version']} ({e['build_type']}), commit {e['git_commit'] or '?'}, built {e['build_date']}"),
        ("Package", e["package"]),
        ("libEinsums", e["libEinsums"] or "(not found)"),
        ("BLAS", _blas_row(e)),
        ("Enabled", " ".join(k.removeprefix("EINSUMS_WITH_") for k, on in e["options"].items() if on) or "-"),
        ("Disabled", " ".join(k.removeprefix("EINSUMS_WITH_") for k, on in e["options"].items() if not on) or "-"),
        ("Python", f"{py['version']} ({py['executable']})"),
        (
            "Optional",
            ", ".join(f"{m} {d['version']}" if d["version"] else f"{m} missing ({d['for']})" for m, d in py["optional"].items()),
        ),
        ("Platform", info["platform"]["system"]),
        ("CPU", f"{cpu.get('model', cpu['machine'])}, {cpu['logical_cores']} logical cores"),
    ]
    if "x86_64_level" in cpu:
        rows.append(("ISA level", f"x86-64-{cpu['x86_64_level']} (the highest SIMD rung this CPU supports)"))
    if cpu["simd_override"]:
        rows.append(("SIMD override", " ".join(f"{k}={v}" for k, v in cpu["simd_override"].items())))
    profile = info["profile"]
    summary = profile["state"]
    if profile.get("path"):
        summary += f", {profile['path']}"
    if profile.get("reason"):
        summary += f" ({profile['reason']})"
    elif profile["state"] == "none":
        summary += " (run einsums tune to measure one)"
    rows.append(("Profile", summary))
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value}")
    return 0
