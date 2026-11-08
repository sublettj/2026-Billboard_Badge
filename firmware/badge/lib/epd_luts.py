"""Short waveform tables for the e-paper fast refresh.

Ported byte for byte from firmware/esp32s3-itc97/src/epd97_lut.h and the
defaults in include/epd97_config.h, where every number was measured on this
panel with a camera. firmware/tests/test_epd_fast.py checks this module
against the C header, so change the C side and re-run the test, or change both.

The table is written to register 0xB0 in the one window the COG accepts it:
after the last 0xA7 load strobe (which loads the panel's own OTP waveform into
the same registers) and before the DC/DC soft start. A table is a set of per
pixel-class tables selected by (red bit << 1) | black bit: 0 paper, 1 ink,
2 red, 3 unreachable, 4 VCOM.

A level byte holds four phases, two bits each, phase 0 in the top bits:
0 ground, 1 VDH, 2 VDL, 3 VDHR (the reduced positive rail red uses).
"""

# --- E2969JS084, "driver 8": 180 bytes, 5 tables x 9 groups x 4 bytes -------
# 103 frames at 32 ms, 3.3 s, in a single trigger. Group 0 is a balanced shake
# (as much VDH as VDL, so no net charge builds up and the paper does not go
# pink), group 1 the chopped drive, groups 2-8 one-frame placeholders: a group
# left at zero is not empty, it runs about four seconds.
_LUT8_PAPER = (0x66, 0x33, 0x33, 0x02, 0x88, 0x31, 0x31, 0x09) + (0x00, 0x10, 0x00, 0x01) * 7
_LUT8_INK = (0x99, 0x33, 0x33, 0x02, 0x44, 0x31, 0x31, 0x09) + (0x00, 0x10, 0x00, 0x01) * 7
_LUT8_VCOM = (0x00, 0x33, 0x33, 0x02, 0x00, 0x31, 0x31, 0x09) + (0x00, 0x10, 0x00, 0x01) * 7

# On the 084 the table sets the duration, so one trigger is the whole frame.
PULLS_084 = 1


def lut_084():
    """Monochrome table: paper, then ink for tables 1-3, then VCOM."""
    return bytes(_LUT8_PAPER + _LUT8_INK * 3 + _LUT8_VCOM)


# --- E2969JS0B1, "driver B": 400 bytes, 5 tables x 10 groups x 8 bytes ------
# Measured on a conditioned film: ink L* 15.1, paper L* 86.9, contrast 71.8,
# in 3460 ms against 33400 ms for the panel's own waveform. On this COG each
# 0x15 trigger runs a fixed ~1070 ms whatever the table asks for, so the dose
# is delivered as three pulls of the latched table.
PULLS_B1 = 3

_B_INK_TABLE = 1
_B_PAP_TABLE = 0
_B_RED_TABLE = 2
_B_SHAKE_GROUPS = 1
_B_SHAKE_F = 10
_B_SHAKE_REP = 3
_B_INK_SHAKE = 0x66       # VDH,VDL,VDH,VDL
_B_PAP_SHAKE = 0x99       # the reverse, so the two halves are pushed apart
_B_RED_SHAKE = 0xEE
_B_DRIVE_GROUPS = 4
_B_DRIVE_ON = 45
_B_DRIVE_OFF = 1
_B_DRIVE_REP = 3
_B_INK_LEVEL = 0x44       # VDH darkens
_B_WHITE_LEVEL = 0xAA     # VDL on all four phases
_B_PAP_ON = 45
_B_PAP_OFF = 3            # paper keeps a longer rest: full duty cost brightness
_B_RED_LEVEL = 0xFF
_B_RED_GROUPS = 9
_B_RED_WHITE_GROUPS = 1   # clear before setting
_B_RED_WHITE_LEVEL = 0xAA
_B_PH2 = 1


def _put_group(out, q, level, on, off, rep):
    out[q] = level
    out[q + 1] = on
    out[q + 2] = off
    out[q + 3] = _B_PH2
    out[q + 4] = on
    out[q + 5] = off
    out[q + 6] = _B_PH2
    out[q + 7] = rep


def _red_level(g):
    rg = g - _B_SHAKE_GROUPS if g >= _B_SHAKE_GROUPS else 0
    return _B_RED_WHITE_LEVEL if rg < _B_RED_WHITE_GROUPS else _B_RED_LEVEL


def lut_b1():
    """epd_build_lutb() from epd97_lut.h."""
    out = bytearray(400)
    for t in range(5):
        for g in range(10):
            q = (t * 10 + g) * 8
            if t == 4:                                   # VCOM stays at ground
                _put_group(out, q, 0x00, 1, 1, 1)
                continue
            if g < _B_SHAKE_GROUPS:                      # the split shake
                if t == _B_INK_TABLE:
                    shk = _B_INK_SHAKE
                elif t == _B_PAP_TABLE:
                    shk = _B_PAP_SHAKE
                elif t == _B_RED_TABLE:
                    shk = _B_RED_SHAKE
                else:
                    shk = 0x00
                _put_group(out, q, shk, _B_SHAKE_F, _B_SHAKE_F, _B_SHAKE_REP)
                continue
            if g < _B_SHAKE_GROUPS + _B_DRIVE_GROUPS:    # the main drive
                on, off = _B_DRIVE_ON, _B_DRIVE_OFF
                if t == _B_INK_TABLE:
                    lvl = _B_INK_LEVEL
                elif t == _B_RED_TABLE:
                    lvl = _red_level(g)
                elif t == _B_PAP_TABLE:
                    lvl = _B_WHITE_LEVEL
                    on, off = _B_PAP_ON, _B_PAP_OFF
                else:
                    lvl = 0x00
                _put_group(out, q, lvl, on, off, _B_DRIVE_REP)
                continue
            if t == _B_RED_TABLE and g < _B_SHAKE_GROUPS + _B_RED_GROUPS:
                # red keeps driving where black and white have finished
                _put_group(out, q, _red_level(g), _B_DRIVE_ON, _B_DRIVE_OFF, _B_DRIVE_REP)
                continue
            _put_group(out, q, 0x00, 1, 1, 1)
    return bytes(out)
