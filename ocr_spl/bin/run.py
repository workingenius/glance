# encoding=utf8

import os
import cv2 as cv
from argparse import ArgumentParser
from ocr_spl.model.model import SimpleCharOcr

def get_args():
    parser = ArgumentParser(description='image hash similar')
    parser.add_argument('--mode', type=str, default='build', help='build/query')
    parser.add_argument('--hash', type=str, default='hHash', help='dHash/aHash/pHash/hHash/d2Hash')
    parser.add_argument('--char_images', type=str, default='data/char_images.txt', help='query image')
    parser.add_argument('--save_to', type=str, default='data/model.pkl', help='save model')
    parser.add_argument('--load_from', type=str, default='data/model.pkl', help='load model')
    parser.add_argument('--image', type=str, default='data/1.jpg', help='query image')
    parser.add_argument('--dist', type=int, default='3', help='sim distance')

    args = parser.parse_args()
    return args


def run(args):
    socr = SimpleCharOcr()

    # 构建库
    if args.mode == 'build':
        socr.hash_type = args.hash
        assert os.path.exists(args.char_images)
        char_imgs_list = []
        for line in open(args.char_images):
            # char : image_file
            char_imgs_list.append((line[0], line[1:].strip()))
        print("Image Numbers: {}".format(len(char_imgs_list)))

        # 依次读取image文件
        char_img_mat = []
        for c, imgfile in char_imgs_list:
            # import pdb; pdb.set_trace()
            img = cv.imread(imgfile, cv.IMREAD_GRAYSCALE)
            char_img_mat.append([c, img])

        socr.build(char_img_mat)

        # 保存模型
        socr.save(args.save_to)

    # 检索
    if args.mode == 'query':
        assert os.path.exists(args.load_from) and os.path.isfile(args.load_from)
        socr.load(args.load_from)

        # 读取args.image进行检索, 假定了输入图片为白底黑字
        img = cv.imread(args.image, cv.IMREAD_GRAYSCALE)
        char = socr.search(img, dist=args.dist)
        print('image file: {} \t char: {}'.format(args.image, char))


if __name__ == '__main__':
    # init db

    args = get_args()
    run(args)
