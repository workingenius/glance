from typing import List
import math
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


def img_color_border(img, color=np.array([0, 255, 0])):
    """
    给图片边框染上颜色

    color 用 np.array 表示，长度为 3，分别表示 rgb  默认染为绿色
    """
    def bw2rgb(i):
        """给黑白图加维度"""
        if len(img.shape) == 2:
            i2 = i[:, :, np.newaxis]
            return np.concatenate([i2, i2, i2], axis=2)
        elif len(img.shape) == 3:
            return img.copy()
        else:
            raise ValueError('invalid image')

    img = bw2rgb(img)
    img[0, :] = color
    img[-1, :] = color
    img[:, 0] = color
    img[:, -1] = color
    return img


# lines that is cut from a whole image

def img_lines_rebuild(img_lst):
    """用友好的方式展示被分割成行的图片"""
    width = None
    for img in img_lst:
        # 所有图片应同宽
        assert width is None or width == img.shape[1]
        if width is None:
            width = img.shape[1]

    il = []
    for img in img_lst:
        img = img_color_border(img)
        il.append(img)

    ir = np.concatenate(il, axis=0)
    return ir


def img_lines_cut_chars(img_lst):
    """切好行的图片，每一行切出字符"""
    ll = [Line(i) for i in img_lst]
    start, cell_wid = ll_analyze(ll)

    lines = []
    for l in ll:
        img_char_lst = l.cut(start=start, cell_width=cell_wid)
        lines.append(img_char_lst)
    return lines


def ll_analyze(line_lst: List['Line']) -> (int, int):
    """从各行中分析出统一的 字符起始位置 和 格子宽度"""
    char_wid = Line.guess_char_width(line_lst)  # char width 典型的字符宽度，比格子宽度小

    # 筛选出参与搜索的行
    #   空行不参与
    #   有宽字符的（一般是带中文的注释）不参与
    normal_line_lst = [
        l for l in line_lst
        if not l.is_blank
           and not l.has_wide_char(wider_than=char_wid * 1.5)  # 认为中文字符至少比英文字符宽 1.5 倍
    ]
    assert normal_line_lst, 'no normal line found'

    found = False
    phase = None
    cycle = None
    # 开始搜索 起始位置 和 格子宽度
    cast0 = normal_line_lst[0].cast
    other = [l.cast for l in normal_line_lst[1:]]
    for ph, cy in cast_possible_pc(cast0):
        # 如果对第一行适用的 周期相位 对其它普通也适用，则找到
        if all(cast_pc_fits(c, phase=ph, cycle=cy) for c in other):
            phase, cycle = ph, cy
            found = True
            break
    
    assert found, 'phase and cycle for all lines are not found'
    return phase, cycle


def img_chars_rebuild(img_lst_lst):
    """切好字符的图片，拼成原图，看起来更友好"""

    def rebuild_line(img_lst):
        """从字符拼成行"""
        img_lst = [img_color_border(i) for i in img_lst]
        res = np.concatenate(img_lst, axis=1)
        return res

    return img_lines_rebuild([rebuild_line(il) for il in img_lst_lst])


class Line(object):
    """切好的一行"""

    def __init__(self, img):
        self.img = img  # np.ndarray
        self.cast = self.calc_char_cast(img)  # np.array
        self.span_lst = self.calc_spans(self.cast)  # np.ndarray shape=(*, 2)
        self.span_v_lst = (self.span_lst[:, 1] - self.span_lst[:, 0])

    @staticmethod
    def guess_char_width(line_lst: List['Line']) -> int:
        """猜一个典型的字符宽度"""
        # 取所有字符的中位数作为猜测的字符宽度
        #   一般中文字符（宽字符）的数量和标点的数量（窄字符）都比较少
        #   所以掐头去尾之后，其余的就是一般英文字符内容占用的宽度
        all_span_lst = []
        for l in line_lst:
            all_span_lst.append(l.span_v_lst)
        all_span = np.concatenate(all_span_lst)
        return int(np.median(all_span))

    class Cutter(object):
        """cut 方法的辅助类，记录切分的状态"""

        def __init__(self, img):
            self.img = img
            self.width = img.shape[1]
            self.pos = 0
            self.char_img_lst = []

        def cut_at(self, pos):
            assert self.pos < pos <= self.width
            char_img = self.img[:, range(self.pos, pos)]
            self.char_img_lst.append(char_img)
            self.pos = pos

    def cut(self, start, cell_width):
        """根据行起始位置 和 窄格子宽度 切出字符"""
        assert start < cell_width

        # import pdb; pdb.set_trace()
        cutter = self.Cutter(self.img)

        if start > 0:
            cutter.cut_at(start)

        # 开始交替前进框定字符，如果是空行就跳过这个步骤
        done = not len(self.span_lst)
        cw = cell_width
        cur_si = 0  # cur span index
        while not done:
            sp_st = self.span_lst[cur_si][0]
            sp_en = self.span_lst[cur_si][1]
            ce_en = cutter.pos + cw

            if ce_en <= sp_st:  # 空白字符
                cutter.cut_at(ce_en)
            elif sp_st < ce_en <= sp_en:  # 交叠，产生字符
                cutter.cut_at(sp_en)
            elif sp_en < ce_en:  # 合并多个图像到一个字符
                while sp_en < ce_en:
                    cur_si += 1
                    if cur_si < len(self.span_lst):
                        sp_st = self.span_lst[cur_si][0]
                        sp_en = self.span_lst[cur_si][1]
                    else:
                        if sp_en > cutter.pos:
                            cutter.cut_at(sp_en)
                        done = True
                        break

        if cutter.pos < self.img.shape[1]:
            cutter.cut_at(self.img.shape[1])

        char_img_lst = cutter.char_img_lst

        # 切分之后宽度不应发生变化
        after_cut = sum([i.shape[1] for i in char_img_lst])
        before_cut = self.img.shape[1]
        assert after_cut == before_cut, 'line width changed after cut: {} != {}'.format(after_cut, before_cut)

        return char_img_lst

    def has_wide_char(self, wider_than) -> bool:
        """
        检查本行是否有宽字符
        
        标准是，比 wider_than 指定的阈值更宽
        """
        return (self.span_v_lst >= wider_than).any()

    @staticmethod
    def calc_char_cast(img_line):
        """从 一行的图片 生成 字符投射"""
        return img_line.sum(axis=0)

    @staticmethod
    def calc_spans(line_cast):
        """根据 字符投射 找到内容所在的片段"""
        return cal_ups_and_downs(line_cast)

    @property
    def is_blank(self):
        return (self.cast == 0).all()

# cast related functions


def cast_show(cast):
    """打开新窗口展示 cast"""
    plt.plot(cast)
    plt.show()


def cast_analyze(cast):
    """根据投射分析相位和周期"""
    for ph, cy in cast_possible_pc(cast):
        return ph, cy
    else:
        assert False, 'possible phase and cycle not found'


def cast_possible_pc(cast):
    """所有对 cast 来说可行的 相位和周期"""
    assert cast.dtype == np.uint64
    ud = cal_ups_and_downs(cast)
    spans = (ud[:, 1] - ud[:, 0])
    md = np.median(spans)
    md = int(md)

    # 遍历 所有 周期和相位
    for phace in range(0, md * 2):
        for cycle in range(md, md * 2):
            if cast_pc_fits(cast, phace, cycle):
                yield phace, cycle


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


def cal_ups_and_downs(cast):
    """根据 投射计算上行和下行，一般来说，上行和下行都是内容的边沿"""

    one = np.array([0])

    rc = np.int64(cast)
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
