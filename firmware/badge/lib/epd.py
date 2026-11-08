"""E-Paper Display Driver for Billboard Badge

Ported from Wiring in https://github.com/PervasiveDisplays/EPD_Driver_GU_large/blob/e408154652802a82e520cc25ba04adeb347d0528/src/EPD_Driver.cpp

--COPYRIGHT--
Brief The drivers and functions of development board
Copyright (c) 2012-2021 Pervasive Displays Inc. All rights reserved.
 Authors: Pervasive Displays Inc.
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import asyncio as aio
import btree
import framebuf
import gc
from machine import Pin, SoftSPI
from micropython import const
import sys
import time

import lvgl

from lib.epd_luts import PULLS_084, PULLS_B1, lut_084, lut_b1
from lib.epd_pack import nonzero, pack_plain, pack_rev
from lib.large_png import show_png

EPD_CS1_PIN = Pin(13, mode=Pin.OUT)
EPD_CS2_PIN = Pin(3, mode=Pin.OUT)
EPD_BUSY1_PIN = Pin(12, mode=Pin.IN)
EPD_BUSY2_PIN = Pin(8, mode=Pin.IN)
EPD_DC_PIN = Pin(10, mode=Pin.OUT)  # Data High Command Low
EPD_RST_PIN = Pin(11, mode=Pin.OUT)
EPD_SCL_PIN = Pin(21, mode=Pin.OUT)
EPD_SDA_PIN = Pin(47, mode=Pin.OUT)
EPD_SPI_MISO_PIN = Pin(45, mode=Pin.IN)  # Not connected, signal not needed
EPD_PWR_EN_PIN = 48


pdi_cp = 0x9608
pdi_size = 0x96  # 9.69"
display_width = const(960)
half_width = const(display_width // 2)
display_height = const(672)
# Frame-buffer size = width * height / 8 * depth, uint32_t
# image_data_size is really only half the panel pixels, of one color
half_frame_one_color_size = const(display_height * display_width // (8 * 2))
# The width of rows in the full-frame and half-frame framebuffers
# Counting bytes, not pixels, since each byte is 8 packed pixels
display_width_bytes = const(display_width // 8)
display_half_width_bytes = const(display_width_bytes // 2)

# --- Fast refresh --------------------------------------------------------------
# display.update(fast=True) runs a short black-and-white waveform instead of the
# panel's own: ~3.5 s a frame on the B1 instead of ~35 s. The mechanism is the
# one firmware/esp32s3-itc97 measured on this panel: the short table is latched
# once per session, the COG is then kept powered and later frames are only the
# plane data and a bare 0x15 trigger.
#
# The short waveform cannot make red, and red already on the film survives it,
# so: red pixels are drawn as black on a fast frame, the first fast frame after
# anything put red on the panel runs the full waveform instead, and every
# FAST_FRAMES_PER_GLOBAL-th fast frame does too, which is what keeps ghosting
# from building up. Those promoted frames still draw red as black.
FAST_FRAMES_PER_GLOBAL = 10
SESSION_IDLE_MS = 20_000     # power the COG down after this long without a frame
FAST_MAX_FAILURES = 3        # give up on fast refresh until reboot after this many
T_OFF_MIN_MS = 1500          # rails have to bleed down between COG cycles
BUSY_START_MS = 2000         # BUSY must go low this soon after 0x15
BUSY_DONE_MS = 180_000       # ceiling for any refresh, the 084's own is ~51 s
READY_WAIT_MS = 10_000       # B1 holds BUSY low while its booster ramps
SECOND_PASS_MS = 1000        # some panels drop BUSY again for a second pass


# Two displays use different bit orders
def reverse_byte(byte: int) -> int:
    byte = (byte & 0xF0) >> 4 | (byte & 0x0F) << 4
    byte = (byte & 0xCC) >> 2 | (byte & 0x33) << 2
    byte = (byte & 0xAA) >> 1 | (byte & 0x55) << 1
    return byte


class EPaperDisplay:
    width = display_width
    height = display_height

    def __init__(self) -> None:
        print("Initializing E-Paper Display")
        EPD_DC_PIN.value(True)
        EPD_RST_PIN.value(True)
        EPD_CS1_PIN.value(True)
        EPD_CS2_PIN.value(True)
        self.spi = SoftSPI(
            baudrate=8_000_000, sck=EPD_SCL_PIN, mosi=EPD_SDA_PIN, miso=EPD_SPI_MISO_PIN
        )
        self.spi.init()

        try:
            with open("/data/hwconf", "rb") as f:
                hwconf = btree.open(f)
                display_pn = hwconf.get(b"display_pn", b"B1")
                if display_pn == b"84":
                    self._reverse_fb_bits = False
                else:
                    self._reverse_fb_bits = True
        except OSError:
            self._reverse_fb_bits = True
        self._panel_084 = not self._reverse_fb_bits
        self._fast_lut = lut_084() if self._panel_084 else lut_b1()
        self._fast_pulls = PULLS_084 if self._panel_084 else PULLS_B1

        # Fast-refresh session state. The panel's contents are unknown at boot
        # (the splash, or whatever the last firmware left), so treat it as
        # carrying red: the first fast request then runs the full waveform.
        self._fast_requested = False
        self._session_open = False
        self._session_last_ms = 0
        self._panel_has_red = True
        self._fast_since_global = 0
        self._fast_failures = 0
        self._off_ms = None
        self.last_refresh = None   # (mode, frame_ms, pull_ms_list) for telemetry

        # The framebuffers are set and display should be updated.
        self._framebuffer_ready = aio.Event()

        """Each byte is a mini-row of 8 bits for pixels of one color (1 sets the color, 0 is white)
        Bytes are in rows across the screen, as seen on the badge
        0 ------------------------> 59 [60 bytes per row]
        60 ----------------------> 119
        ...                        ...
        [672 rows of 60 bytes per row]
        """
        self._framebuffer_black = bytearray(half_frame_one_color_size * 2)
        self._framebuffer_red = bytearray(half_frame_one_color_size * 2)
        # Framebuffers that can be drawn to directly
        self.black = framebuf.FrameBuffer(
            self._framebuffer_black, display_width, display_height, framebuf.MONO_HLSB
        )
        self.red = framebuf.FrameBuffer(
            self._framebuffer_red, display_width, display_height, framebuf.MONO_HLSB
        )

        # Is the display done updating? Start set since the display is not updating before boot.
        self._display_updated = aio.Event()
        self._display_updated.set()
        gc.collect()

        # Set up lvgl
        lvgl.init()
        # Give LVGL a buffer 2-bytes-per-pixel for RGB565 format
        self._lv_draw_buf = bytearray(display_width * display_height * 2)

        self.lv_display = lvgl.display_create(display_width, display_height)
        self.lv_display.set_color_format(lvgl.COLOR_FORMAT.RGB565)
        self.lv_display.set_buffers(
            self._lv_draw_buf, None, len(self._lv_draw_buf),
            lvgl.DISPLAY_RENDER_MODE.PARTIAL,
        )
        self.lv_display.set_flush_cb(self._lv_flush)
        self.lv_screen = lvgl.screen_active()
        self.lv_screen.set_size(display_width, display_height)

    def _lv_flush(self, disp, area, color_p):
        print("Calling EPaperDisplay._lv_flush")
        x1, y1 = area.x1, area.y1
        w = area.x2 - x1 + 1
        h = area.y2 - y1 + 1
        try:
            stride = lvgl.draw_buf_width_to_stride(w, lvgl.COLOR_FORMAT.RGB565)
        except AttributeError:
            stride = w * 2
        src = color_p.__dereference__(stride * h)
        width = display_width
        for row in range(h):
            base = (y1 + row) * width + x1
            row_off = row * stride
            for col in range(w):
                i = row_off + col * 2
                c = src[i] | (src[i + 1] << 8)  # RGB565, little-endian
                r = (c >> 11) & 0x1F
                g = (c >> 5) & 0x3F
                b = c & 0x1F
                idx = base + col
                byte = idx >> 3
                mask = 1 << (7 - (idx & 7))
                clear = mask ^ 0xFF
                if (r & 0x10) and not (g & 0x20) and not (b & 0x10):
                    self._framebuffer_red[byte] |= mask
                    self._framebuffer_black[byte] &= clear
                elif not (r & 0x10) and not (g & 0x20) and not (b & 0x10):
                    self._framebuffer_black[byte] |= mask
                    self._framebuffer_red[byte] &= clear
                else:
                    self._framebuffer_black[byte] &= clear
                    self._framebuffer_red[byte] &= clear
        disp.flush_ready()

    def update(self, fast=False):
        """Done updating the framebuffer, update the display with the contents.

        fast=False (the default) runs the panel's own waveform: full quality,
        red included, ~35 s. fast=True runs the short black-and-white waveform,
        ~3.5 s, with red drawn as black; see FAST_FRAMES_PER_GLOBAL above.
        """
        self._fast_requested = fast
        self._framebuffer_ready.set()

    def clear(self):
        """Clear framebuffers to full white"""
        self.black.fill(0)
        self.red.fill(0)

    def is_done_updating(self):
        """Is the display done updating?"""
        return self._display_updated.is_set()

    def draw_image(self, image, height: int, width: int, row: int = 0, col: int = 0):
        for img_r in range(height):
            for img_c in range(width):
                color = image[img_r][img_c]
                red = color[0] > 127
                black = color[0] < 128 and color[1] < 128 and color[2] < 128
                self.black.pixel((col + img_c), (row + img_r), int(black))
                self.red.pixel((col + img_c), (row + img_r), int(red))

    def draw_png(self, filename: str, row: int = 0, col: int = 0):

        gc.collect()
        print(f"Loading {filename} This will take a while... {time.time()}")
        show_png(filename, self._framebuffer_black, self._framebuffer_red, row, col)
        print("Done loading png")
        gc.collect()

    async def run(self):
        self._bl = bytearray(half_frame_one_color_size)
        self._br = bytearray(half_frame_one_color_size)
        self._rl = bytearray(half_frame_one_color_size)
        self._rr = bytearray(half_frame_one_color_size)
        self._zero_half = bytes(half_frame_one_color_size)
        while True:
            if self._framebuffer_ready.is_set():
                self._display_updated.clear()
                try:
                    await self._refresh(self._fast_requested)
                except Exception as ex:
                    # Keep the display task alive: a dead task means a badge
                    # that never draws again.
                    print("epd: refresh raised")
                    sys.print_exception(ex)
                    try:
                        await self._session_close()
                    except Exception as ex2:
                        sys.print_exception(ex2)
                self._framebuffer_ready.clear()
                gc.collect()
                self._display_updated.set()
            elif self._session_open and time.ticks_diff(
                    time.ticks_ms(), self._session_last_ms) > SESSION_IDLE_MS:
                print("epd: fast session idle, powering the COG down")
                await self._session_close()
            await aio.sleep_ms(20)

    # --------------------------------------------------------------- refresh
    def _pack(self, red_as_ink):
        """Split the framebuffers into the per-COG halves the panel takes.

        Returns (black_left, red_left, black_right, red_right). With
        red_as_ink the red plane is OR'ed into black and the red planes go out
        empty, which is how a monochrome frame shows red content.
        """
        pack = pack_rev if self._reverse_fb_bits else pack_plain
        bk = self._framebuffer_black
        rd = self._framebuffer_red
        if red_as_ink:
            pack(bk, rd, self._bl, self._br)
            return self._bl, self._zero_half, self._br, self._zero_half
        pack(bk, bk, self._bl, self._br)
        pack(rd, rd, self._rl, self._rr)
        return self._bl, self._rl, self._br, self._rr

    async def _refresh(self, fast):
        t_frame = time.ticks_ms()
        has_red = nonzero(self._framebuffer_red, len(self._framebuffer_red))
        drive_fast = fast
        why = ""
        if fast and self._fast_failures >= FAST_MAX_FAILURES:
            drive_fast = False
            why = " (fast refresh disabled after failures)"
        elif fast and self._panel_has_red:
            drive_fast = False
            why = " (promoted: red on the panel)"
        elif fast and self._fast_since_global >= FAST_FRAMES_PER_GLOBAL - 1:
            drive_fast = False
            why = " (promoted: periodic clean)"

        # The full waveform needs a fresh COG cycle, and a latched short
        # waveform cannot be swapped inside a session.
        if self._session_open and not drive_fast:
            await self._session_close()

        # A caller that asked for fast expects monochrome, promoted or not.
        halves = self._pack(red_as_ink=fast)
        pulls = None
        if drive_fast:
            pulls = await self._fast_update(*halves)
            if pulls is None:
                self._fast_failures += 1
                print(f"epd: fast refresh failed ({self._fast_failures}/{FAST_MAX_FAILURES}),"
                      " redrawing on the full waveform")
                await self._session_close()
                drive_fast = False
                why = " (fallback)"
            else:
                self._fast_failures = 0
        if drive_fast:
            self._fast_since_global += 1
            self._panel_has_red = False
        else:
            await self.global_update(*halves)
            self._fast_since_global = 0
            self._panel_has_red = bool(has_red) and not fast

        mode = "fast" if drive_fast else "full"
        frame_ms = time.ticks_diff(time.ticks_ms(), t_frame)
        self.last_refresh = (mode, frame_ms, pulls)
        print(f"epd: {mode} refresh{why} {frame_ms} ms"
              + (f", pulls {pulls} ms" if pulls else ""))

    async def _fast_update(self, black_left, red_left, black_right, red_right):
        """One fast frame. Returns the per-pull BUSY-low times, or None on failure."""
        pulls = self._fast_pulls
        times = []
        if not self._session_open:
            await self._reset()
            self._send_images(black_left, red_left, black_right, red_right)
            await self.cog_initial()
            # The one window for 0xB0: after the last 0xA7 strobe in
            # cog_initial, before the soft start. Both chip selects are low,
            # so both COGs latch the same table.
            self._send_index_data(0xB0, self._fast_lut)
            await self._dcdc_soft_start()
            self._session_open = True
            if not await self._wait_idle(READY_WAIT_MS):
                print("epd: booster not ready after 10 s, triggering anyway")
            ms = await self._trigger(second_pass=True)
            if ms < 0:
                return None
            times.append(ms)
            pulls -= 1
        else:
            # Session held: the rails, the table and the registers are all
            # still loaded. A frame is the plane data and the triggers.
            self._send_images(black_left, red_left, black_right, red_right)
        for _ in range(pulls):
            ms = await self._trigger()
            if ms < 0:
                return None
            times.append(ms)
        self._session_last_ms = time.ticks_ms()
        return times

    def _busy_idle(self):
        # BUSY is active low on both COGs.
        return EPD_BUSY1_PIN.value() and EPD_BUSY2_PIN.value()

    async def _wait_idle(self, timeout_ms):
        t0 = time.ticks_ms()
        while not self._busy_idle():
            if time.ticks_diff(time.ticks_ms(), t0) > timeout_ms:
                return False
            await aio.sleep_ms(10)
        return True

    async def _trigger(self, second_pass=False):
        """Send 0x15 and wait the refresh out. Returns BUSY-low ms, or <0."""
        self._send_index_data(0x15, [0x3C])
        t0 = time.ticks_ms()
        # BUSY drops within milliseconds of the trigger. Spin for it rather
        # than await, so a busy asyncio neighbour cannot make us miss the edge.
        while self._busy_idle():
            if time.ticks_diff(time.ticks_ms(), t0) > BUSY_START_MS:
                print("epd: BUSY never went low after 0x15")
                return -2
            time.sleep_ms(1)
        t1 = time.ticks_ms()
        if not await self._wait_idle(BUSY_DONE_MS):
            print("epd: refresh never finished")
            return -3
        busy_ms = time.ticks_diff(time.ticks_ms(), t1)
        if second_pass:
            t2 = time.ticks_ms()
            while time.ticks_diff(time.ticks_ms(), t2) < SECOND_PASS_MS:
                if not self._busy_idle():
                    await self._wait_idle(BUSY_DONE_MS)
                    busy_ms = time.ticks_diff(time.ticks_ms(), t1)
                    print("epd: the panel ran a second waveform pass")
                    break
                await aio.sleep_ms(10)
        return busy_ms

    async def _session_close(self):
        if not self._session_open:
            return
        self._session_open = False
        await self._dcdc_soft_shutdown()

    async def global_update(
        self,
        black_left: bytearray | bytes | memoryview[int],
        red_left: bytearray | bytes | memoryview[int],
        black_right: bytearray | bytes | memoryview[int],
        red_right: bytearray | bytes | memoryview[int],
    ):
        print("Updating display to new image")
        await self._reset()
        tick = time.ticks_ms()
        self._send_images(black_left, red_left, black_right, red_right)
        tock = time.ticks_ms()
        print(tock-tick)
        
        print("display working")
        tick = time.ticks_ms()
        await self.cog_initial()
        await self._dcdc_soft_start()
        await self._display_refresh()
        await self._dcdc_soft_shutdown()
        print("Done updating display")
        tock = time.ticks_ms()
        print(tock-tick)


    async def cog_initial(self):
        self._send_index_data(0x05, [0x7D])
        await aio.sleep_ms(200)
        self._send_index_data(0x05, [0x00])
        await aio.sleep_ms(10)
        self._send_index_data(0xD8, [0x80])  # MS_SYNC)
        self._send_index_data(0xD6, [0x00])  # BVSS
        self._send_index_data(0xA7, [0x10])
        await aio.sleep_ms(100)
        self._send_index_data(0xA7, [0x00])
        await aio.sleep_ms(100)
        self._send_index_data(0x03, [0x00, 0x11])  # OSC

        # Display 1
        self._send_index_data_left(0x44, [0x00])
        self._send_index_data_left(0x45, [0x80])
        self._send_index_data_left(0xA7, [0x10])
        await aio.sleep_ms(100)
        self._send_index_data_left(0xA7, [0x00])
        await aio.sleep_ms(100)
        self._send_index_data_left(0x44, [0x06])
        self._send_index_data_left(0x45, [0x82])  # Temperature 0x82@25C
        self._send_index_data_left(0xA7, [0x10])
        await aio.sleep_ms(100)
        self._send_index_data_left(0xA7, [0x00])
        await aio.sleep_ms(100)

        # Display 2
        self._send_index_data_right(0x44, [0x00])
        self._send_index_data_right(0x45, [0x80])
        self._send_index_data_right(0xA7, [0x10])
        await aio.sleep_ms(100)
        self._send_index_data_right(0xA7, [0x00])
        await aio.sleep_ms(100)
        self._send_index_data_right(0x44, [0x06])
        self._send_index_data_right(0x45, [0x82])  # Temperature 0x82@25C
        self._send_index_data_right(0xA7, [0x10])
        await aio.sleep_ms(100)
        self._send_index_data_right(0xA7, [0x00])
        await aio.sleep_ms(100)

        self._send_index_data(0x60, [0x25])  # TCON
        self._send_index_data_left(0x61, [0x01])  # STV_DIR for Display 1
        self._send_index_data(0x02, [0x00])  # VCOM

    async def cog_poweroff(self):
        """CoG shutdown function
        Shuts down the CoG and DC/DC circuit after all update functions
        """
        self._send_index_data(0x09, [0x7F, 0x7D, 0x00])
        await aio.sleep_ms(200)
        while not EPD_BUSY1_PIN.value() or not EPD_BUSY2_PIN.value():
            await aio.sleep_ms(1)
        EPD_DC_PIN.value(False)
        EPD_CS1_PIN.value(False)
        await aio.sleep_ms(150)
        EPD_RST_PIN.value(False)

    def _send_index_data(self, index: int, data: bytes | list):
        EPD_DC_PIN.value(False)
        EPD_CS1_PIN.value(False)
        EPD_CS2_PIN.value(False)
        self.spi.write(bytes([index]))
        EPD_CS1_PIN.value(True)
        EPD_CS2_PIN.value(True)
        EPD_DC_PIN.value(True)
        EPD_CS1_PIN.value(False)
        EPD_CS2_PIN.value(False)
        self.spi.write(bytes(data))
        EPD_CS1_PIN.value(True)
        EPD_CS2_PIN.value(True)

    def _send_index_data_left(self, index: int, data: bytes | list | memoryview[int]):
        EPD_DC_PIN.value(False)
        EPD_CS1_PIN.value(False)
        EPD_CS2_PIN.value(True)
        self.spi.write(bytes([index]))
        EPD_CS1_PIN.value(True)
        EPD_DC_PIN.value(True)
        EPD_CS1_PIN.value(False)
        self.spi.write(bytes(data))
        EPD_CS1_PIN.value(True)

    def _send_index_data_right(self, index: int, data: bytes | list | memoryview[int]):
        EPD_DC_PIN.value(False)
        EPD_CS1_PIN.value(True)
        EPD_CS2_PIN.value(False)
        self.spi.write(bytes([index]))
        EPD_CS2_PIN.value(True)
        EPD_DC_PIN.value(True)
        EPD_CS2_PIN.value(False)
        self.spi.write(bytes(data))
        EPD_CS2_PIN.value(True)

    async def _display_refresh(self):
        while not EPD_BUSY1_PIN.value() or not EPD_BUSY2_PIN.value():
            await aio.sleep_ms(100)
        self._send_index_data(0x15, [0x3C])
        await aio.sleep_ms(5)

    async def _reset(self):
        """CoG driver power-on hard reset"""
        if self._off_ms is not None:
            gap = time.ticks_diff(time.ticks_ms(), self._off_ms)
            if gap < T_OFF_MIN_MS:
                await aio.sleep_ms(T_OFF_MIN_MS - gap)
        await aio.sleep_ms(200)
        EPD_RST_PIN.value(True)
        await aio.sleep_ms(20)
        EPD_RST_PIN.value(False)
        await aio.sleep_ms(200)
        EPD_RST_PIN.value(True)
        await aio.sleep_ms(200)
        EPD_CS1_PIN.value(True)
        EPD_CS2_PIN.value(True)
        await aio.sleep_ms(5)

    def _send_images(
        self,
        black_left: bytearray | bytes | memoryview[int],
        red_left: bytearray | bytes | memoryview[int],
        black_right: bytearray | bytes | memoryview[int],
        red_right: bytearray | bytes | memoryview[int],
    ):
        assert len(black_left) == half_frame_one_color_size
        assert len(red_left) == half_frame_one_color_size
        assert len(black_right) == half_frame_one_color_size
        assert len(red_right) == half_frame_one_color_size
        # Bit order is already right: _pack() reverses for the B1 on the way in.

        # DUW for both left and right halves
        self._send_index_data(0x13, [0x00, 0x3B, 0x00, 0x00, 0x9F, 0x02])
        # DRFW for Both left and right halves
        self._send_index_data(0x90, [0x00, 0x3B, 0x00, 0xA9])
        ram_rw_data = [0x3B, 0x00, 0x14]
        self._send_index_data_left(0x12, ram_rw_data)  # RAM_RW for left half
        self._send_index_data_left(0x10, black_left)  # Black frame for left half
        self._send_index_data_left(0x12, ram_rw_data)  # RAM_RW for left half
        self._send_index_data_left(0x11, red_left)  # Red frame for left half
        self._send_index_data_right(0x12, ram_rw_data)  # RAM_RW for right half
        self._send_index_data_right(0x10, black_right)  # black frame for right half
        self._send_index_data_right(0x12, ram_rw_data)  # RAM_RW for right half
        self._send_index_data_right(0x11, red_right)  # Red frame for right half


    async def _dcdc_soft_start(self):
        """DC-DC soft-start command
        Implemented after image data are uploaded to CoG
        Specific to mid-sized EPDs only
        """
        self._send_index_data(0x51, [0x50, 0x01])

        for val in range(1, 5):
            self._send_index_data(0x09, [0x1F])
            self._send_index_data(0x51, [0x50, val])
            self._send_index_data(0x09, [0x9F])
            await aio.sleep_ms(2)

        for val in range(1, 11):
            self._send_index_data(0x09, [0x1F])
            self._send_index_data(0x51, [0x0A, val])
            self._send_index_data(0x09, [0x9F])
            await aio.sleep_ms(2)

        for val in range(3, 11):
            self._send_index_data(0x09, [0x7F])
            self._send_index_data(0x51, [0x0A, val])
            self._send_index_data(0x09, [0xFF])
            await aio.sleep_ms(2)

        for val in range(9, 1, -1):
            self._send_index_data(0x09, [0x7F])
            self._send_index_data(0x51, [val, 0x0A])
            self._send_index_data(0x09, [0xFF])
            await aio.sleep_ms(2)

        self._send_index_data(0x09, [0xFF])
        await aio.sleep_ms(10)

    async def _dcdc_soft_shutdown(self):
        """DC-DC soft-shutdown command
        Implemented after image data are uploaded to CoG
        Specific to mid-sized EPDs only
        """
        if not await self._wait_idle(BUSY_DONE_MS):
            print("epd: BUSY stuck before shutdown, shutting down anyway")
        self._send_index_data(0x09, [0x7F])
        self._send_index_data(0x05, [0x7D])
        self._send_index_data(0x09, [0x00])
        await aio.sleep_ms(200)
        await self._wait_idle(BUSY_DONE_MS)
        EPD_DC_PIN.value(False)
        EPD_CS1_PIN.value(False)
        EPD_RST_PIN.value(False)
        EPD_CS1_PIN.value(True)
        self._off_ms = time.ticks_ms()


display = EPaperDisplay()
