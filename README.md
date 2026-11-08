![Billboard_Badge](assets/frame_graphic_demo.png "2026 Hackaday Supercon Billboard Badge")

# Badge Hardware
The badge uses a 9.7 inch Pervasive Displays E2969JS0B1 electronic paper panel that was recycled from price tags which were destined for a now bankrupt German department store chain, coupled with an ESP32-S3 microcontroller.

## Pins

The board only has two truly free GPIOs

* GPIO9
* GPIO41

The SAO ports, however, have many possibilities

* SAO1: GPIO2 and GPIO42
* SAO2: GPIO4 and GPIO5
* SAO3: GPIO7 and GPIO6
* I2C bus: SDA on GPIO17 and SCL on GPIO18

The row of WS2812-compatible LEDs is on GPIO38

## Buttons

* Menu +/-: GPIO40 / GPIO15
* User A/B: GPIO39 / GPIO15

# Applications
In its out of the box configuration the badge can show the 2026 Supercon schedule, act as a digital name tag, and cycle through a built-in collection of art from Hackaday's incredible Art Director Joe Kim. If you come up with additional apps that you'd like to share, feel free to make a pull request.

## Stuck: a language model inside the badge

`apps/stuck.py` runs a small transformer language model on the badge itself, trained from scratch to be a mind that woke up inside this badge and keeps asking where it is. The model, the training pipeline and the int8 C runtime are in [`llm/`](llm/README.md); the MicroPython binding is `firmware/user_c_modules/stuck_llm/`.

# Displaying Images

The badge's display only supports three colors: white, black, red. Shading between those needs to be implemented by dithering pixels of those colors together. Unlike LCD, CRT, or other displays you may be used to, you cannot set multiple colors within a single pixel, each pixel is only one of the three colors.

The badge firmware picks a color for each pixel based on this logic:
- If the red value is >50% and blue and green are both <50%, then the pixel will be red.
- If the red, green, and blue values of the pixel are all <50%, then the pixel will be black.
- Otherwise the pixel will be white.

Additionally, since the display is so large, loading large images can easily run out of memory. Therefore, it is best to compress each image down to a limited color space before placing on the badge. This is easily done with [ImageMagick](https://imagemagick.org/), namely with the command `magick oldfile.png -remap scripts/palette.gif newfile.png`. This reduces the image down to a colorspace that only supports 3 colors, which are white, black, and red.

Based on the above pixel logic, where each pixel is already one of the three supported colors, it's easy to see how the image will be displayed. Luckily, but converting the image on your computer, you can see what it will look like before rendering it on the badge.

## Fast refresh

`display.update()` runs the panel's own waveform: full quality and red, but about 35 seconds. `display.update(fast=True)` runs a short black-and-white waveform instead, about 3.5 seconds a frame, for apps that change the screen often. Red pixels are drawn as black on a fast frame. The driver keeps the panel's controller powered between fast frames, and on its own runs the full waveform every tenth fast frame and for the first fast frame after anything red was shown, which is what keeps ghosting in check. The waveform tables come from the C driver in `firmware/esp32s3-itc97`; `python3 firmware/tests/test_epd_fast.py` checks the port and the refresh policy on a PC.
