#!/usr/bin/env python3
"""Map a CUDA / nvidia-smi GPU index to the EGL device index MuJoCo expects.

MuJoCo's EGL backend selects its render GPU with ``MUJOCO_EGL_DEVICE_ID``, which is an
index into ``eglQueryDevicesEXT`` — and that enumeration order does **not** match
nvidia-smi / CUDA ordinals, nor is it steered by ``CUDA_VISIBLE_DEVICES`` (e.g. on this
box nvidia-smi GPU 3 is EGL device 0). So to co-locate the sim client's renderer on the
same physical GPU as the policy server, we must translate the index.

Mapping strategy (ctypes + libEGL, no PyOpenGL):
  1. Enumerate EGL devices; keep those advertising ``EGL_NV_device_cuda`` (the real GPUs).
  2. Read each device's DRM node via ``EGL_DRM_DEVICE_FILE_EXT`` (e.g. /dev/dri/card1).
  3. Resolve the DRM node to a PCI bus id through /sys/class/drm/<card>/device.
  4. Match that PCI id against ``nvidia-smi --query-gpu=index,pci.bus_id`` to get the
     nvidia-smi GPU index, yielding ``{gpu_index: egl_index}``.

(The more direct ``EGL_CUDA_DEVICE_NV`` attribute query returns EGL_BAD_ATTRIBUTE on this
driver, so PCI matching is used instead — it's driver-version-independent.)

Usage:
    python egl_device_map.py            # print the full {gpu: egl} mapping
    python egl_device_map.py 3          # print MUJOCO_EGL_DEVICE_ID for nvidia-smi GPU 3

On any failure it prints the requested index unchanged (safe fallback) and warns on
stderr, so the eval still launches rather than crashing.
"""

import ctypes
import os
import subprocess
import sys

EGL_EXTENSIONS = 0x3055
EGL_DRM_DEVICE_FILE_EXT = 0x3233


def _pci_tail(bus_id: str) -> str:
    """Normalize a PCI bus id to a comparable 'dddd:bb:dd.f' tail (handles nvidia-smi's
    8-hex domain vs sysfs's 4-hex domain)."""
    return bus_id.strip().lower()[-12:]


def _drm_to_pci(devnode: str) -> str | None:
    """/dev/dri/cardN -> normalized PCI bus id via sysfs, or None."""
    card = os.path.basename(devnode)
    try:
        target = os.path.realpath(f"/sys/class/drm/{card}/device")
    except OSError:
        return None
    return _pci_tail(os.path.basename(target))


def gpu_to_egl_map() -> dict[int, int]:
    """Return {nvidia_smi_gpu_index: egl_device_index}."""
    egl = ctypes.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]

    def _proc(name, restype, argtypes):
        addr = egl.eglGetProcAddress(name.encode())
        if not addr:
            raise RuntimeError(f"EGL extension function {name} unavailable")
        return ctypes.CFUNCTYPE(restype, *argtypes)(addr)

    EGLDeviceEXT = ctypes.c_void_p
    eglQueryDevicesEXT = _proc(
        "eglQueryDevicesEXT",
        ctypes.c_uint,
        [ctypes.c_int, ctypes.POINTER(EGLDeviceEXT), ctypes.POINTER(ctypes.c_int)],
    )
    eglQueryDeviceStringEXT = _proc("eglQueryDeviceStringEXT", ctypes.c_char_p, [EGLDeviceEXT, ctypes.c_int])

    n = ctypes.c_int(0)
    if not eglQueryDevicesEXT(0, None, ctypes.byref(n)) or n.value <= 0:
        raise RuntimeError("eglQueryDevicesEXT returned no devices")
    devs = (EGLDeviceEXT * n.value)()
    got = ctypes.c_int(0)
    if not eglQueryDevicesEXT(n.value, devs, ctypes.byref(got)):
        raise RuntimeError("eglQueryDevicesEXT enumeration failed")

    # nvidia-smi: pci bus id -> gpu index
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"], text=True)
    pci_to_gpu: dict[str, int] = {}
    for line in out.strip().splitlines():
        idx, bus = (x.strip() for x in line.split(","))
        pci_to_gpu[_pci_tail(bus)] = int(idx)

    mapping: dict[int, int] = {}
    for egl_idx in range(got.value):
        exts = eglQueryDeviceStringEXT(devs[egl_idx], EGL_EXTENSIONS)
        if not exts or b"EGL_NV_device_cuda" not in exts:
            continue  # skip non-NVIDIA / duplicate (e.g. Mesa) EGL devices
        drm = eglQueryDeviceStringEXT(devs[egl_idx], EGL_DRM_DEVICE_FILE_EXT)
        if not drm:
            continue
        pci = _drm_to_pci(drm.decode())
        gpu = pci_to_gpu.get(pci) if pci else None
        if gpu is not None and gpu not in mapping:
            mapping[gpu] = egl_idx
    return mapping


def main() -> None:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else None
    try:
        mapping = gpu_to_egl_map()
    except Exception as e:
        if target is None:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"warn: EGL probe failed ({e}); falling back to MUJOCO_EGL_DEVICE_ID={target}", file=sys.stderr)
        print(target)
        return

    if target is None:
        print({f"gpu{g}": f"egl{e}" for g, e in sorted(mapping.items())})
        return
    if target in mapping:
        print(mapping[target])
    else:
        print(f"warn: nvidia-smi GPU {target} not found in EGL map ({mapping}); using {target}", file=sys.stderr)
        print(target)


if __name__ == "__main__":
    main()
