import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


# image related function


def img_read(path):
    """读入图片"""
    img = Image.open(path)
    return np.array(img)


def img_write(img, path):
    """图片写到文件"""
    i = Image.fromarray(img)
    i.save(path)


def img_bw(img):
    """将 rgb 图片处理为黑白"""
    assert len(img.shape) == 3
    assert img.dtype == np.uint8

    # 如果图片有 ALPHA 层，去掉它
    if img.shape[2] == 4:
        img = img[:, :, (0, 1, 2)]

    # 转为灰度图像
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
    img_gray = rgb2gray(img)

    # TODO 目前暂时假设图片为黑色背景，其余情况后续考虑

    # 二值化，背景为
    bw_img = ((img_gray > 100) * 255)
    bw_img = np.uint8(bw_img)
    return bw_img


def img_rowcast(img):
    assert len(img.shape) == 2
    assert img.dtype == np.uint8
    return img.sum(axis=1)


def img_cut_rows(img):
    """把图片切成行"""
    rc = img_rowcast(img)
    start, rowhei = cast_analyze(rc)
    assert start < rowhei

    sub_img_lst = []
    sub_img_lst.append(
        img[range(0, start)]
    )
    cur = start + rowhei
    while cur <= len(img):
        sub_img_lst.append(
            img[range(cur, min(cur + rowhei, len(img)))]
        )
        cur += rowhei

    return sub_img_lst


# rows that is cut from a whole image

def img_rows_rebuild(img_lst):
    """用友好的方式展示被分割成行的图片"""
    width = None
    for img in img_lst:
        assert len(img.shape) == 2
        # 所有图片应同宽
        assert width is None or width == img.shape[1]
        if width is None:
            width = img.shape[1]

    def bw2rgb(i):
        """给黑白图加维度"""
        i2 = i[:, :, np.newaxis]
        return np.concatenate([i2, i2, i2], axis=2)

    il = []
    for img in img_lst:
        img = bw2rgb(img)
        # 边缘做红
        red = np.array([255, 0, 0])
        img[0, :] = red
        img[-1, :] = red
        img[:, 0] = red
        img[:, -1] = red
        il.append(img)

    ir = np.concatenate(il, axis=0)
    return ir

# cat related functions


def cast_show(cast):
    """打开新窗口展示 cast"""
    plt.plot(cast)
    plt.show()


def cast_analyze(rowcast):
    """根据行投射分析 起始行位置 和 行高"""
    rc = rowcast
    assert rc.dtype == np.uint64
    ud = cal_ups_and_downs(rowcast)
    spans = (ud[:, 1] - ud[:, 0])
    md = np.median(spans)
    md = int(md)

    phace = None  # 相位，即起始行位置
    cycle = None  # 周期，即行高
    found = False
    # 暴力搜索周期和相位
    for phace in range(0, md * 2):
        for cycle in range(md, md * 2):
            if cast_pc_fits(rc, phace, cycle):
                found = True
                break
        if found:
            break
    
    if not found:
        raise ValueError('this case is not considered')

    return phace, cycle


def cast_pc_fits(cast, phase, cycle):
    """
    检查在 cast 情况下，phase 和 cylce 是恰当的分割方法
    
    一个恰当的分割方法，不应碰到任何内容
    """
    borders = np.arange(phase, len(cast), cycle)

    a1 = borders  # 边缘后一像素
    b1 = borders - 1  # 边缘前一像素
    b1[b1 < 0] = 0

    av = cast[a1]  # 前一像素投射值
    bv = cast[b1]  # 后一像素投射值
    bor_val = np.column_stack((av, bv))

    # 若边缘前后某一像素投射值为 0，就算边缘有效
    return ((bor_val[:, 0] == 0) | (bor_val[:, 1] == 0)).all()


def cal_ups_and_downs(rowcast):
    """根据行投射计算上行和下行，一般来说，上行和下行处都是行的边沿"""

    one = np.array([0])

    rc = np.int64(rowcast)
    rc_right1 = np.concatenate((one, rc[:-1]))

    change = np.column_stack((rc_right1, rc))

    width = len(rc)
    zero_th = width // 1000  # 几乎为 0 的阈值
    change_th = width // 10  # 发生突变的阈值
    assert change_th > zero_th

    ups = np.where(
        (change[:, 0] < zero_th)  # 起始位置几乎为 0
        & ((change[:, 1] - change[:, 0]) > change_th)  # 发生突变
    )[0]
    downs = np.where(
        (change[:, 1] < zero_th)
        & ((change[:, 1] - change[:, 0]) < -change_th)
    )[0]

    # 一般情况下，上升和下降数量应该相等
    assert len(ups) == len(downs)
    # 一般情况下，每个上升小于每个下降
    assert (ups < downs).all()

    spans = np.column_stack((ups, downs))
    return spans
