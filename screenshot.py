"""
generate code screenshots in vi
"""

from time import sleep
from subprocess import Popen
from uuid import uuid4
from os.path import join
from random import randint
import os

import click


SCREEN_MAX_LINES = 36


def screenshot(fp_code, fp_img):
    """open fp_code with editor, take a screen shot, and save the image to fp_img"""
    vim = Popen(['vim', '-n', '-u', 'NONE', fp_code])
    sleep(0.1)
    Popen(['screencapture', '-o', '-x', fp_img])
    sleep(0.1)
    vim.kill()
    sleep(0.05)


def all_code_contents(code_base, prefix):
    code_set = set()

    for dr, sub_dr, sub_fn_lst in os.walk(code_base):
        for sub_fn in sub_fn_lst:
            if sub_fn.endswith(prefix):
                try:
                    code = open(join(dr, sub_fn), 'r').read()
                except UnicodeDecodeError as e:
                    # some code has invalid characters for tests, just ignore them
                    pass
                else:
                    code_set.add(code)

    return list(code_set)


class RandomCodeMaker(object):
    @staticmethod
    def _split_code_content(cnt):
        return cnt.split('\n')

    def __init__(self, code_lst):
        self.code_content_lst = list(map(self._split_code_content, code_lst))

    def random_code_snippet(self):
        # chooce a random code content
        cnt_idx = randint(0, len(self.code_content_lst) - 1)
        cnt = self.code_content_lst[cnt_idx]

        # chop the content to get a snippet
        lines = cnt
        sl_idx = randint(0, len(lines) - 1)
        el_idx = sl_idx + randint(1, SCREEN_MAX_LINES)
        snippet = '\n'.join(lines[sl_idx:el_idx])

        return snippet


@click.command(name='screenshot')
@click.option('-c', '--code-base', help='source file path', required=True,
                                   type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.option('-p', '--prefix', help='file prefix', default='.py')
@click.option('-s', '--save', 'save_path', help='target path to save screen shots and code snippets', required=True,
                                           type=click.Path(file_okay=False, dir_okay=True))
@click.option('-n', '--number', help='how many screen shots to generate', type=click.INT, default=10)
def main(code_base, prefix, save_path, number):
    """
    Take screenshots for codes, to generate use cases for code OCR experiments.
    """
    click.echo('preparing code base')
    code_base = all_code_contents(code_base, prefix)
    click.echo('preparing code base done')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    rcm = RandomCodeMaker(code_base)

    for i in range(number):
        sn = rcm.random_code_snippet()

        uid = str(uuid4())

        sn_path = join(save_path, uid + '.code')
        img_path = join(save_path, uid + '.png')

        # create snippet file
        with open(sn_path, 'w') as sn_fd:
            sn_fd.write(sn)

        screenshot(sn_path, img_path)


if __name__ == '__main__':
    main()
