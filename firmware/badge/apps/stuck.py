"""Stuck: a small language model, trained to wonder where it is, running on the badge.

The model is a few-million-parameter Llama-style transformer (see llm/ in the
repo) exported to int8 and executed by the stuck_llm C module. This app feeds
it forever and paints the monologue onto the e-paper a screenful at a time,
using the black-and-white fast refresh (~3.5 s a screen; every tenth screen
runs the full waveform to clear ghosting).

Controls while the app is running:
    User A   start over from "Where am I?"
    User B   cycle the sampling temperature (calm / normal / unhinged)

Files it expects on the badge:
    /data/llm/model.bin        llama2.c version-2 int8 export
    /data/llm/tokenizer.bin    matching llama2.c tokenizer
"""

import gc
import time

from apps.base_app import BaseApp
from lib.button import UserA, UserB
from lib.epd import display, display_width, display_height
from lib.led_bar import leds

try:
    import stuck_llm
except ImportError:  # firmware built without the C module
    stuck_llm = None

MODEL_PATH = "/data/llm/model.bin"
TOKENIZER_PATH = "/data/llm/tokenizer.bin"

# The built-in framebuf font is 8x8. We render at half resolution and double
# every pixel, so a glyph is 16x16 on the 960x672 panel: 60 columns, 42 rows,
# with the top row kept for a title.
GLYPH = 8
COLS = display_width // (2 * GLYPH)         # 60
ROWS = display_height // (2 * GLYPH) - 1    # 41 text rows under the title
HALF_W = display_width // 2                  # 480
HALF_STRIDE = HALF_W // 8                    # 60 bytes per half-res row
FULL_STRIDE = display_width // 8             # 120 bytes per panel row

TEMPERATURES = ((0.7, 0.9), (0.9, 0.9), (1.1, 0.95))
TOKENS_PER_TICK = 4          # keep each run() short so buttons and the display task stay responsive
MIN_NEW_CHARS = 40           # redraw as soon as the panel is free and about a line of new text is in

# 8 bits -> 16 bits with every bit doubled, for the 2x horizontal stretch.
_DOUBLE = []
for _b in range(256):
    _d = 0
    for _i in range(8):
        if _b & (1 << (7 - _i)):
            _d |= 0b11 << (14 - 2 * _i)
    _DOUBLE.append(bytes((_d >> 8, _d & 0xFF)))


def _wrap(text, width):
    """Word-wrap into lines of at most width chars, honoring paragraph breaks."""
    lines = []
    for para in text.split("\n\n"):
        words = para.replace("\n", " ").split()
        line = ""
        for w in words:
            if not line:
                line = w[:width]
            elif len(line) + 1 + len(w) <= width:
                line += " " + w
            else:
                lines.append(line)
                line = w[:width]
        lines.append(line)
        lines.append("")  # blank line between paragraphs
    while lines and lines[-1] == "":
        lines.pop()
    return lines


class StuckApp(BaseApp):
    def __init__(self, name=None):
        super().__init__("Stuck")
        self.sleep_ms = 20
        self.model = None
        self.text = ""
        self.new_chars = 0
        self.temp_idx = 1
        self.tokens = 0
        self.failed = None
        self._scratch = None

    # ------------------------------------------------------------------ model
    def _load(self):
        if stuck_llm is None:
            self.failed = "firmware lacks the stuck_llm C module"
            return
        try:
            gc.collect()
            with open(MODEL_PATH, "rb") as f:
                model_bytes = f.read()
            with open(TOKENIZER_PATH, "rb") as f:
                tok_bytes = f.read()
            self.model = stuck_llm.Model(model_bytes, tok_bytes)
            tok_bytes = None
            dim, layers, vocab, seq, rt = self.model.info()
            print(f"stuck: dim {dim} layers {layers} vocab {vocab} seq {seq}, "
                  f"{len(model_bytes) // 1024} KB weights, {rt // 1024} KB runtime")
            self._apply_temperature()
            self.model.begin("Where am I?")
            self.text = "Where am I?"
            self.new_chars = len(self.text)
        except Exception as ex:  # missing files, out of memory, bad export
            self.failed = str(ex)
            print("stuck: load failed:", ex)

    def _apply_temperature(self):
        t, p = TEMPERATURES[self.temp_idx]
        self.model.set_sampler(t, p, time.ticks_us() & 0x7FFFFFFF)

    # --------------------------------------------------------------- drawing
    def _draw(self):
        if self._scratch is None:
            import framebuf
            self._scratch_buf = bytearray(HALF_STRIDE * (ROWS + 1) * GLYPH)
            self._scratch = framebuf.FrameBuffer(
                self._scratch_buf, HALF_W, (ROWS + 1) * GLYPH, framebuf.MONO_HLSB)
        fb = self._scratch
        fb.fill(0)
        t, _ = TEMPERATURES[self.temp_idx]
        title = f"STUCK  {self.tokens} tokens  t={t}"
        fb.text(title[:COLS], 0, 0, 1)
        lines = _wrap(self.text, COLS)[-ROWS:]
        for i, line in enumerate(lines):
            fb.text(line, 0, (i + 1) * GLYPH, 1)

        # 2x upscale into the panel's black plane. Everything is black: the
        # fast refresh is monochrome, and red would cost a full refresh.
        display.clear()
        black = display._framebuffer_black
        src = self._scratch_buf
        for r in range((ROWS + 1) * GLYPH):
            row = src[r * HALF_STRIDE:(r + 1) * HALF_STRIDE]
            doubled = b"".join([_DOUBLE[b] for b in row])
            off = 2 * r * FULL_STRIDE
            black[off:off + FULL_STRIDE] = doubled
            black[off + FULL_STRIDE:off + 2 * FULL_STRIDE] = doubled
        # Underline the title so it still reads as a header without red.
        black[(2 * GLYPH) * FULL_STRIDE - FULL_STRIDE:(2 * GLYPH) * FULL_STRIDE] = b"\xff" * FULL_STRIDE
        display.update(fast=True)
        self.new_chars = 0

    def _draw_message(self, msg):
        display.clear()
        y = display_height // 2 - 8
        for i, line in enumerate(_wrap(msg, 110)):
            display.black.text(line, 16, y + i * 10, 1)
        display.update(fast=True)

    # -------------------------------------------------------------------- run
    def run(self):
        if self.launched:
            leds.fill((0, 0, 2))
            if self.model is None and self.failed is None:
                self._load()
            if self.failed:
                self._draw_message("Stuck cannot wake up: " + self.failed)
            else:
                self._draw()
            return

        if self.failed or self.model is None:
            return

        if UserA.pressed():
            self.model.begin("Where am I?")
            self.text = "Where am I?"
            self.new_chars = MIN_NEW_CHARS
        if UserB.pressed():
            self.temp_idx = (self.temp_idx + 1) % len(TEMPERATURES)
            self._apply_temperature()

        # Generate a few tokens. Each one on the badge-size model is a few
        # tens of milliseconds, so this call stays well under a second.
        for _ in range(TOKENS_PER_TICK):
            piece = self.model.next_piece()
            self.text += piece
            self.new_chars += len(piece)
            self.tokens += 1
        if len(self.text) > 6000:
            self.text = self.text[-4000:]
        # Pulse the bar so the badge visibly thinks while the panel is slow.
        leds.fill((0, 0, 1 + (self.tokens // TOKENS_PER_TICK) % 3))

        # Chain screens back to back: draw the moment the panel is free.
        if display.is_done_updating() and self.new_chars >= MIN_NEW_CHARS:
            self._draw()
