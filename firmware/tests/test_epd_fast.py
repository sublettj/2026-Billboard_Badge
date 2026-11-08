#!/usr/bin/env python3
"""Host test for the badge's e-paper fast refresh (CPython, no hardware).

Runs the real firmware/badge/lib/epd.py under CPython with the MicroPython-only
modules stubbed, against a fake panel that records every register write and
pulls BUSY low for a while after each 0x15 trigger. It checks:

  1. the short waveform tables in lib/epd_luts.py match the C driver's
     epd97_lut.h byte for byte (compiles a tiny C program; needs cc)
  2. the viper packers in lib/epd_pack.py split, OR and bit-reverse the planes
     exactly as the old pure-Python path did
  3. the refresh policy and register sequencing: promotion to the full
     waveform when red is on the panel and every tenth frame, 0xB0 written in
     its one window, three pulls on the B1, bare triggers inside a held
     session, red drawn as ink, idle shutdown, and fallback after failures

    python3 firmware/tests/test_epd_fast.py
"""

import asyncio
import builtins
import os
import subprocess
import sys
import tempfile
import time
import traceback
import types
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
BADGE = os.path.normpath(os.path.join(HERE, "..", "badge"))
CDRV = os.path.normpath(os.path.join(HERE, "..", "esp32s3-itc97"))

# ----------------------------------------------------------------------------
# MicroPython shims
# ----------------------------------------------------------------------------
SLEEP_SCALE = 1 / 25  # run every asyncio sleep in the driver 25x faster


def install_shims():
    mp = types.ModuleType("micropython")
    mp.const = lambda x: x
    mp.viper = lambda f: f
    mp.native = lambda f: f
    sys.modules["micropython"] = mp
    builtins.ptr8 = lambda b: b

    time.ticks_ms = lambda: int(time.monotonic() * 1000)
    time.ticks_us = lambda: int(time.monotonic() * 1_000_000)
    time.ticks_diff = lambda a, b: a - b
    time.sleep_ms = lambda ms: time.sleep(ms / 1000)
    asyncio.sleep_ms = lambda ms: asyncio.sleep(ms / 1000 * SLEEP_SCALE)
    sys.print_exception = lambda e: traceback.print_exception(e)

    machine = types.ModuleType("machine")

    class Pin:
        OUT, IN = 1, 0

        def __init__(self, *a, **k):
            self._v = 1

        def value(self, v=None):
            if v is None:
                return self._v
            self._v = int(bool(v))

    class SoftSPI:
        def __init__(self, *a, **k):
            pass

        def init(self):
            pass

        def write(self, b):
            pass

    machine.Pin, machine.SoftSPI = Pin, SoftSPI
    sys.modules["machine"] = machine

    framebuf = types.ModuleType("framebuf")
    framebuf.MONO_HLSB = 0

    class FrameBuffer:
        def __init__(self, buf, w, h, fmt):
            self.buf, self.w, self.h = buf, w, h

        def fill(self, c):
            v = 0xFF if c else 0
            for i in range(len(self.buf)):
                self.buf[i] = v

        def pixel(self, x, y, c):
            i = (y * self.w + x) >> 3
            m = 0x80 >> (x & 7)
            self.buf[i] = (self.buf[i] | m) if c else (self.buf[i] & ~m & 0xFF)

    framebuf.FrameBuffer = FrameBuffer
    sys.modules["framebuf"] = framebuf
    sys.modules["btree"] = types.ModuleType("btree")
    sys.modules["lvgl"] = mock.MagicMock()
    lp = types.ModuleType("lib.large_png")
    lp.show_png = lambda *a, **k: None
    sys.modules["lib.large_png"] = lp
    sys.path.insert(0, BADGE)


# ----------------------------------------------------------------------------
# 1. LUTs against the C header
# ----------------------------------------------------------------------------
C_DUMP = r"""
#include <stdio.h>
#include "epd97_lut.h"
int main(void) {
  uint8_t b[LUTB_BYTES];
  epd_build_lutb(b);
  for (unsigned i = 0; i < LUT8_BYTES; i++) printf("%02x", k_lut8_base[i]);
  printf("\n");
  for (unsigned i = 0; i < LUTB_BYTES; i++) printf("%02x", b[i]);
  printf("\n");
  return 0;
}
"""


def test_luts():
    from lib.epd_luts import lut_084, lut_b1
    with tempfile.TemporaryDirectory() as d:
        src, exe = os.path.join(d, "dump.c"), os.path.join(d, "dump")
        with open(src, "w") as f:
            f.write(C_DUMP)
        subprocess.run(["cc", "-std=c11", "-I", os.path.join(CDRV, "src"),
                        "-I", os.path.join(CDRV, "include"), "-o", exe, src], check=True)
        out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout.split()
    assert lut_084().hex() == out[0], "084 table differs from epd97_lut.h"
    assert lut_b1().hex() == out[1], "B1 table differs from epd97_lut.h"
    assert len(lut_084()) == 180 and len(lut_b1()) == 400
    print("ok  LUTs match epd97_lut.h (180 B driver 8, 400 B driver B)")


# ----------------------------------------------------------------------------
# 2. packers against the old pure-Python path
# ----------------------------------------------------------------------------
def reverse_byte(b):
    b = (b & 0xF0) >> 4 | (b & 0x0F) << 4
    b = (b & 0xCC) >> 2 | (b & 0x33) << 2
    return (b & 0xAA) >> 1 | (b & 0x55) << 1


def ref_halves(fb, rev):
    left, right = bytearray(), bytearray()
    for r in range(672):
        left += fb[r * 120:r * 120 + 60]
        right += fb[r * 120 + 60:(r + 1) * 120]
    if rev:
        left = bytearray(reverse_byte(b) for b in left)
        right = bytearray(reverse_byte(b) for b in right)
    return bytes(left), bytes(right)


def test_pack():
    import random
    from lib.epd_pack import nonzero, pack_plain, pack_rev
    rng = random.Random(1)
    bk = bytearray(rng.getrandbits(8) for _ in range(80640))
    rd = bytearray(rng.getrandbits(8) & rng.getrandbits(8) for _ in range(80640))
    for fn, rev in ((pack_rev, True), (pack_plain, False)):
        L, R = bytearray(40320), bytearray(40320)
        fn(bk, bk, L, R)
        assert (bytes(L), bytes(R)) == ref_halves(bk, rev)
        fn(bk, rd, L, R)
        merged = bytes(a | b for a, b in zip(bk, rd))
        assert (bytes(L), bytes(R)) == ref_halves(merged, rev)
    z = bytearray(80640)
    assert nonzero(z, len(z)) == 0
    z[80639] = 4
    assert nonzero(z, len(z)) == 1
    print("ok  packers match the old split + bit-reverse path, red OR'ed as ink")


# ----------------------------------------------------------------------------
# 3. refresh policy and sequencing against a fake panel
# ----------------------------------------------------------------------------
class FakePanel:
    """Records register writes; BUSY goes low for pull_s after each 0x15."""

    def __init__(self, pull_s=0.05):
        self.pull_s = pull_s
        self.busy_until = 0.0
        self.never_busy = False   # simulate a COG that ignores 0x15
        self.log = []

    def busy_value(self):
        return 0 if time.monotonic() < self.busy_until else 1

    def write(self, side, index, data):
        self.log.append((side, index, bytes(data)))
        if index == 0x15 and not self.never_busy:
            self.busy_until = time.monotonic() + self.pull_s

    def take(self):
        log, self.log = self.log, []
        return log


class Busy:
    def __init__(self, panel):
        self.panel = panel

    def value(self, v=None):
        return self.panel.busy_value()


def idx(log, index, side=None):
    return [i for i, (s, x, _) in enumerate(log) if x == index and (side is None or s == side)]


def data_of(log, index, side):
    return [d for s, x, d in log if x == index and s == side]


def import_epd():
    """Import lib/epd.py with deferred annotations: MicroPython ignores them,
    but CPython would evaluate the driver's memoryview[int] hints and fail."""
    import __future__
    import importlib.util
    path = os.path.join(BADGE, "lib", "epd.py")
    spec = importlib.util.spec_from_file_location("lib.epd", path)
    mod = importlib.util.module_from_spec(spec)
    with open(path) as f:
        code = compile(f.read(), path, "exec",
                       flags=__future__.annotations.compiler_flag, dont_inherit=True)
    sys.modules["lib.epd"] = mod
    exec(code, mod.__dict__)
    return mod


async def run_policy():
    epd = import_epd()
    from lib.epd_luts import lut_b1
    panel = FakePanel()
    d = epd.display
    epd.EPD_BUSY1_PIN = epd.EPD_BUSY2_PIN = Busy(panel)
    d._send_index_data = lambda i, data: panel.write("both", i, data)
    d._send_index_data_left = lambda i, data: panel.write("L", i, data)
    d._send_index_data_right = lambda i, data: panel.write("R", i, data)
    epd.SESSION_IDLE_MS = 400
    epd.BUSY_START_MS = 100
    task = asyncio.create_task(d.run())

    async def frame(fast, black_px=(), red_px=()):
        d.clear()
        for x, y in black_px:
            d.black.pixel(x, y, 1)
        for x, y in red_px:
            d.red.pixel(x, y, 1)
        d.update(fast=fast)
        await asyncio.sleep(0.005)
        while d._framebuffer_ready.is_set() or not d.is_done_updating():
            await asyncio.sleep(0.005)
        return d.last_refresh[0], panel.take()

    def is_shutdown(log):
        return any(x == 0x09 and dat == b"\x00" for _, x, dat in log)

    # boot: panel state unknown, treated as red -> the first fast frame runs full
    mode, log = await frame(True, black_px=[(0, 0)])
    assert mode == "full", mode
    assert not idx(log, 0xB0) and len(idx(log, 0x15)) == 1 and is_shutdown(log)
    print("ok  first fast frame after boot is promoted to the full waveform")

    # second: opens a session, latches 0xB0 in its window, three pulls, no shutdown
    mode, log = await frame(True, black_px=[(0, 0), (480, 0)])
    assert mode == "fast", mode
    b0 = idx(log, 0xB0)
    assert len(b0) == 1 and log[b0[0]][2] == lut_b1(), "0xB0 must carry the 400-byte B1 table once"
    assert max(idx(log, 0xA7, "R")) < b0[0] < min(idx(log, 0x51)), "0xB0 outside its window"
    assert len(idx(log, 0x15)) == 3 and not is_shutdown(log)
    bl, br = data_of(log, 0x10, "L")[0], data_of(log, 0x10, "R")[0]
    assert bl[0] == 0x01 and br[0] == 0x01, "x=0 and x=480 must land bit-reversed in byte 0 of each half"
    assert d._session_open
    print("ok  fast frame opens a session: 0xB0 after the last 0xA7, before the soft start; 3 pulls")

    # third..tenth: held session, bare triggers only
    for n in range(3, 11):
        mode, log = await frame(True, black_px=[(n, n)])
        assert mode == "fast", (n, mode)
        assert not idx(log, 0xA7) and not idx(log, 0xB0) and not idx(log, 0x51), "held frame re-ran init"
        assert len(idx(log, 0x15)) == 3 and idx(log, 0x10, "L") and not is_shutdown(log)
    print("ok  frames 3-10 reuse the session: planes plus three bare 0x15, no init")

    # eleventh: periodic clean -> session closed, full waveform
    mode, log = await frame(True, black_px=[(1, 1)])
    assert mode == "full", mode
    assert is_shutdown(log[:idx(log, 0x10, "L")[0]]), "session must close before the full refresh"
    assert not d._session_open
    print("ok  every tenth frame runs the full waveform, closing the session first")

    # red content on a fast frame is drawn as ink, red plane empty
    mode, log = await frame(True, red_px=[(8, 1)])
    assert mode == "fast", mode
    assert data_of(log, 0x10, "L")[0][61] == 0x01, "red pixel must appear in the black plane"
    assert not any(data_of(log, 0x11, "L")[0]) and not any(data_of(log, 0x11, "R")[0])
    print("ok  red on a fast frame is drawn as black, red plane sent empty")

    # a normal update() with red draws red and marks the panel
    mode, log = await frame(False, red_px=[(8, 1)])
    assert mode == "full" and data_of(log, 0x11, "L")[0][61] == 0x01
    mode, log = await frame(True, black_px=[(2, 2)])
    assert mode == "full", "fast after red must be promoted"
    mode, log = await frame(True, black_px=[(3, 3)])
    assert mode == "fast", mode
    print("ok  full update() keeps red; the next fast frame is promoted once, then fast resumes")

    # idle session powers down
    await asyncio.sleep(0.6)
    log = panel.take()
    assert not d._session_open and is_shutdown(log), "idle session should shut the COG down"
    print("ok  an idle session powers the COG down")

    # a COG that ignores 0x15: fall back to full, then stop trying after 3
    panel.never_busy = True
    for k in range(3):
        mode, log = await frame(True, black_px=[(4, 4)])
        assert mode == "full", (k, mode)
    assert d._fast_failures == 3
    mode, log = await frame(True, black_px=[(5, 5)])
    assert mode == "full" and not idx(log, 0xB0), "fast should be disabled after 3 failures"
    print("ok  failed fast frames fall back to the full waveform; disabled after 3")

    task.cancel()


def main():
    install_shims()
    test_luts()
    test_pack()
    asyncio.run(run_policy())
    print("all fast-refresh tests passed")


if __name__ == "__main__":
    main()
