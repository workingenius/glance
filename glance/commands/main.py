import sys
from os.path import dirname
from glob import glob
import click


@click.command(name='glance')
@click.option('-i', '--image', 'img_pth_ptn_lst', multiple=True, help='image path pattern', required=True)
@click.option('-c', '--code', 'code_pth', help='code recognized save path', type=click.Path(file_okay=True, dir_okay=False))
def main(img_pth_ptn_lst, code_pth):
    """Recognize code snippet in the images"""

    # read images
    img_pth_lst = []
    for ptn in img_pth_ptn_lst:
        ptn_lst = glob(ptn)
        ptn_lst.sort()
        img_pth_lst.extend(ptn_lst)

    if len(img_pth_lst) == 0:
        click.Abort('no image given')

    from glance.codeocr import img_read, recognize

    img_lst = []
    for ip in img_pth_lst:
        img = img_read(ip)
        img_lst.append(img)

    code = recognize(img_lst)

    if code_pth:
        with open(code_pth, 'w') as out_fo:
            print(code, file=out_fo)
    elif code_pth is None or code_pth == '--':
        print(code, file=sys.stdout)
    

if __name__ == '__main__':
    main()
