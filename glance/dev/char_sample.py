"""
Generate images contains only one charactor as training samples
"""


import json
import os
import random
from itertools import product
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import click
from PIL import Image, ImageDraw, ImageFont

__all__ = [
    'CharSample',
    'load_samples'
]


INDEX = 'index.json'


class CharSample(object):
    char = None
    font = None  # font family
    size = None  # font size

    @property
    def image(self) -> Image.Image:
        raise NotImplementedError

    def save(self, image_path) -> 'CharSample2':
        assert image_path
        self.image.save(image_path)
        return CharSample2(self.char, self.font, self.size, image_path)


class CharSample1(CharSample):
    """Sample with image in memory"""

    def __init__(self, char, font, size, image):
        assert isinstance(char, str) and len(char) == 1
        assert isinstance(image, Image.Image)
        self.char = char
        self.font = font  # font family
        self.size = size  # font size
        self._image: Image.Image = image

    @property
    def image(self) -> Image.Image:
        return self._image


class CharSample2(CharSample):
    """Sample with image on disk"""

    def __init__(self, char, font, size, image_path):
        assert isinstance(char, str) and len(char) == 1
        self.char = char
        self.font = font  # font family
        self.size = size  # font size
        self.image_path = str(image_path)

    @property
    def image(self) -> Image.Image:
        return Image.open(self.image_path)

    def to_json(self):
        return {
            'char': self.char,
            'font': self.font,
            'size': self.size,
            'image_path': self.image_path,
        }

    @classmethod
    def from_json(cls, js):
        return cls(
            js['char'],
            js['font'],
            js['size'],
            js['image_path'],
        )


def _generate_sample(char, font, size) -> CharSample:
    """Generate a single sample"""

    # create a background image that is initially big enough
    image = Image.new(mode='L', size=(512, 512))

    # measure char size, and draw it one the background
    draw = ImageDraw.ImageDraw(image)
    _font = ImageFont.truetype(font=font, size=size)
    (wid, hei) = draw.textsize(char, font=_font)
    if wid == 0:
        # blank char like space or tab, width is zero
        wid = hei
    draw.text((0, 0), char, font=_font, fill='white')
    
    # clip the image, making it just fit the char
    image = image.crop((0, 0, wid, hei))

    # randomly extend each borders a little, to simulate a char segment bias
    image = _extend_image(image)
    return CharSample1(char, font, size, image=image)


def _extend_image(image: Image.Image) -> Image.Image:
    """Randomly extend each borders a little"""
    
    def _rand_extend(length, rate):
        return round(random.random() * rate * length)

    (wid, hei) = image.size

    # extending length for 4 directions, left, right, top, bottom
    l = _rand_extend(wid, 0.2)
    r = _rand_extend(wid, 0.2)
    t = _rand_extend(hei, 0.2)
    b = _rand_extend(hei, 0.2)

    # new width and height
    nwid = wid + l + r
    nhei = hei + t + b

    # create a background and put the image on it
    back = Image.new(mode='L', size=(nwid, nhei))
    back.paste(image, box=(l, t))
    return back


def generate_samples(chars, fonts, sizes, n) -> Iterator[CharSample]:
    """Generate n samples for every char, every font, every size"""
    assert isinstance(fonts, list)
    assert isinstance(sizes, list)
    for char, font, size in product(chars, fonts, sizes):
        for _ in range(n):
            yield _generate_sample(char, font, size)


def save_samples(samples: Iterator[CharSample], path) -> int:
    """Save some char samples into a directory, at directory <path>, return how many has been saved"""
    count = 0
    index = []
    path = Path(path)
    for s1 in samples:
        img_path = path / (uuid4().hex + '.png')
        img_path = img_path.absolute()
        s2 = s1.save(img_path)
        index.append(
            s2.to_json()
        )
        count += 1
    with open(path / INDEX, 'w') as fo:
        json.dump(index, fo, indent=2, ensure_ascii=False, sort_keys=True)
    return count


def load_samples(path) -> Iterator[CharSample]:
    """Load char samples from a directory that saved"""
    path = Path(path)
    with open(path / INDEX) as fo:
        index = json.load(fo)
    for js in index:
        yield CharSample2.from_json(js)


@click.command(name='char_sample')
@click.option('-c', '--char', help='The char to generate sample')
@click.option('-C', '--file', help='Generate samples for each unique character in the file',
               type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option('-o', '--out', help='Save samples at the directory',
               type=click.Path(file_okay=False, dir_okay=True))
@click.option('-f', '--font', multiple=True, help='Font families to generate', required=True,
               type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option('-s', '--size', multiple=True, type=click.INT, help='Font sizes to generate', required=True)
@click.option('-n', 'number', type=click.INT, help='Numbers to generate for each class', default=1)
def main(char, file, out, font, size, number):
    """Generate charactor images as training samples"""

    # do some check on output dir
    out = Path(out)
    if not out.exists():
        out.mkdir(parents=True)
    elif (out / INDEX).exists():
        click.secho('The output directory has samples already', fg='red')
        return
    elif os.listdir(out):
        click.secho('The output directory is not empty', fg='red')
        return

    # collect chars
    chars = set()
    if file:
        chars |= set(open(file, 'r').read())
    if char:
        chars.add(char)
    char_lst = sorted(chars)

    # collect sizes
    if len(size) > 1:
        size = list(range(min(size), max(size)))

    samples = generate_samples(chars=char_lst, fonts=list(font), sizes=list(size), n=number)
    count = save_samples(samples, path=out)
    click.secho('{} samples generated'.format(count), fg='green')


if __name__ == '__main__':
    main()
