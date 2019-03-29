# coding: utf-8
"""
    captcha.image
    ~~~~~~~~~~~~~
    Generate Image CAPTCHAs, just the normal image CAPTCHAs you are using.
    https://github.com/lepture/captcha/blob/master/captcha/image.py
"""

import os
import random
from PIL import Image
from PIL import ImageFilter
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype

DEFAULT_FONTS = [os.path.join('./fonts', 'DroidSansMono.ttf')]

table  =  []
for  i  in  range( 256 ):
    table.append( i * 1.97 )

def random_color(start, end, opacity=None):
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return (red, green, blue)
    return (red, green, blue, opacity)

class ImageCaptcha(object):
    """Create an image CAPTCHA.
    Many of the codes are borrowed from wheezy.captcha, with a modification
    for memory and developer friendly.
    ImageCaptcha has one built-in font, DroidSansMono, which is licensed under
    Apache License 2. You should always use your own fonts::
        captcha = ImageCaptcha(fonts=['/path/to/A.ttf', '/path/to/B.ttf'])
    You can put as many fonts as you like. But be aware of your memory, all of
    the fonts are loaded into your memory, so keep them a lot, but not too
    many.
    :param width: The width of the CAPTCHA image.
    :param height: The height of the CAPTCHA image.
    :param fonts: Fonts to be used to generate CAPTCHA images.
    :param font_sizes: Random choose a font size from this parameters.
    """
    def __init__(self, width=60, height=70, fonts=None, font_sizes=None):
        self._width = width
        self._height = height
        self._fonts = fonts or DEFAULT_FONTS
        self._font_sizes = font_sizes or (42, 50, 56)
        self._truefonts = []

    @property
    def truefonts(self):
        if self._truefonts:
            return self._truefonts
        self._truefonts = tuple([
            truetype(n, s)
            for n in self._fonts
            for s in self._font_sizes
        ])
        return self._truefonts

    def create_captcha_image(self, chars, color, background):
        """Create the CAPTCHA image itself.
        :param chars: text to be generated.
        :param color: color of the text.
        :param background: color of the background.
        The color should be a tuple of 3 numbers, such as (0, 255, 255).
        """
        def _draw_character(c):
            image = Image.new('RGB', (self._width, self._height), background)
            draw  = Draw(image)
            font = random.choice(self.truefonts)
            w, h = draw.textsize(c, font=font)
            w = random.randint(w-10, w)
            h = random.randint(h-5, h+1)
            dx = 0
            dy = 0
            im = Image.new('RGBA', (w , h))
            Draw(im).text((dx, dy), c, font=font, fill=color)

            # rotate
            im = im.crop(im.getbbox())
            im = im.rotate(random.uniform(-30, 30), Image.BILINEAR, expand=1)

            return im, w, h

        im, width, height = _draw_character(chars)
        img   = im.convert('L').point(table)
        image = Image.new('RGB', (width, height), background)
        draw  = Draw(image)
        image.paste(im, (0, 0), img)

        #if width > self._width:
        #    image = image.resize((self._width, self._height), resample=PIL.Image.LANCZOS)

        return image

    def generate_image(self, chars):
        """Generate the image of the given characters.
        :param chars: text to be generated.
        """
        background = (255, 255, 255)
        color = random_color(10, 200, random.randint(220, 255))
        im = self.create_captcha_image(chars, color, background)
        im = im.filter(ImageFilter.SMOOTH)

        return im
