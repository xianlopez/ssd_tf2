# Implementation of [SSD](https://arxiv.org/abs/1512.02325) in TensorFlow 2.

I'm running it with Ubuntu 18 and Python 3.6.9, on a GTX 1080Ti GPU and a Ryzen 5 2600 processor. The TensorFlow version is 2.3.1.

Currently the mAP I get with this code is around 0.5, which is lower than the 0.7 reported in the paper, so there's something wrong that I'm doing.

In order to train, you need to download VOC 2007 trainval and test, and VOC2012 trainval. Extract them all to the same VOCdevkit folder.

To train, run
```
python train.py
```
There are some options you can configure, have a look at the file `train.py`.

To evaluate, run
```
python evaluate.py --ckpt_idx <number>
```
Again, you can see the list of options in `evaluate.py`.
