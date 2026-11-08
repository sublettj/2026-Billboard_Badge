"""Native helpers that move framebuffer bytes to the panel's layout.

The panel is driven as two halves, one per driver chip, each 60 bytes of every
120-byte row. The B1 panel also wants each byte bit-reversed. Doing that in
plain Python costs several seconds per frame; these viper loops do it in a few
tens of milliseconds, which is what makes a ~3.5 s fast refresh possible.

Viper functions take at most four arguments, hence two packers instead of a flag.
"""

import micropython

_ROWS = 672
_ROW = 120
_HALF = 60

_rev = bytearray(256)
for _b in range(256):
    _r = 0
    for _i in range(8):
        if _b & (1 << _i):
            _r |= 0x80 >> _i
    _rev[_b] = _r
REV = bytes(_rev)
del _rev


@micropython.viper
def pack_rev(src, orsrc, left, right):
    """left/right halves of (src | orsrc), each byte bit-reversed.

    Pass src as orsrc for a plain copy; pass the red plane to draw red as ink.
    """
    s = ptr8(src)
    o = ptr8(orsrc)
    lp = ptr8(left)
    rp = ptr8(right)
    t = ptr8(REV)
    row = 0
    while row < 672:
        so = row * 120
        do = row * 60
        i = 0
        while i < 60:
            lp[do + i] = t[s[so + i] | o[so + i]]
            rp[do + i] = t[s[so + 60 + i] | o[so + 60 + i]]
            i += 1
        row += 1


@micropython.viper
def pack_plain(src, orsrc, left, right):
    """As pack_rev, without the bit reversal (the 084 panel)."""
    s = ptr8(src)
    o = ptr8(orsrc)
    lp = ptr8(left)
    rp = ptr8(right)
    row = 0
    while row < 672:
        so = row * 120
        do = row * 60
        i = 0
        while i < 60:
            lp[do + i] = s[so + i] | o[so + i]
            rp[do + i] = s[so + 60 + i] | o[so + 60 + i]
            i += 1
        row += 1


@micropython.viper
def nonzero(buf, n: int) -> int:
    """1 if any of the first n bytes of buf is set, else 0."""
    p = ptr8(buf)
    i = 0
    while i < n:
        if p[i]:
            return 1
        i += 1
    return 0
