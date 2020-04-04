# encoding=utf8

import cv2 as cv
from .hashval import ImageHashCal
import pickle
import json
from tqdm import tqdm


class SimpleCharOcr(object):
    def __init__(self, hash_type=None):
        self.scale = [16, 16]
        self._hash_type = hash_type
        self.hash_func = ImageHashCal().hash_func
        self.model_data = {'hash_type': self.hash_type, 'hash_value': {}}

    @property
    def hash_type(self):
        return self._hash_type

    @hash_type.setter
    def hash_type(self, hash):
        self._hash_type = hash

    def save(self, filepath):
        serialization = json.dumps(self.model_data, ensure_ascii=False, indent=2)
        print(serialization)
        pickle.dump(self.model_data, open(filepath, mode='wb'))

    def load(self, filepath):
        self.model_data = pickle.load(open(filepath, mode='rb'))
        self._hash_type = self.model_data.get('hash_type', None)
        assert self.hash_type, 'load %s failed'.format(filepath)

    def image_bin_scale(self, img):
        # 对图片做二值化 和缩放至16*16
        # import pdb; pdb.set_trace()
        img_new = cv.resize(img, tuple(self.scale))
        return img_new

    def build(self, images):
        assert self._hash_type
        self.model_data['hash_type'] = self._hash_type
        imghash = self.hash_func[self.hash_type]
        for char, img in tqdm(images, total=len(images)):
            img = self.image_bin_scale(img)
            hashval = imghash(img)
            assert hashval not in self.model_data['hash_value']
            self.model_data['hash_value'][hashval] = char
        return self.model_data

    def search(self, img, dist=5):
        imghash = self.hash_func[self._hash_type]
        img = self.image_bin_scale(img)
        y = imghash(img)
        # 顺序查询 可优化
        dist_vec = []
        for x, c in self.model_data['hash_value'].items():
            d = ImageHashCal.hanmming_distance(x, y)
            if d <= dist:
                dist_vec.append([d, c])
        res = sorted(dist_vec, key=lambda k: k[0])
        return res[0][1] if len(res) > 0 else None



