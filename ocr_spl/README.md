# 简单的字符ocr

适用于标准光学字符截图
0. 先对图片做二值化，并缩放到标准尺寸;
1. 线下使用字符图片集合作为训练集，计算图片hash建立基础库，这里选用直方图hash；
2. 线上对请求的图片缩放到标准尺寸，计算图片hash，与基础库进行相似计算，找到候选；
3. 通过LM对候选进行选择；

## step
- build : 建库 保存模型
- query: 计算query image hash， 查库 返回结果

## save model
hash_type
hashvalue: char

## 样本数据

假定了图片都是*白底黑字*，
如此，需要前置做图片处理，将背景置白，前景置黑


```
data/images : 存储单字符图片
data/data/char_images.txt : 存储单字符对应的图片文件路径
	format：char filepath
```
## 执行
```
export PYTHONPATH=`pwd`
cd ocr_spl


# build
python bin/run.py

# query
python bin/run.py --mode query --image data/images/7.png --dist=3
```