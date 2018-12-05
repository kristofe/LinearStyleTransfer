## Learning Linear Transformations for Fast Arbitrary Style Transfer
**[[Paper]](https://arxiv.org/abs/1808.04537v1)**

<img src="doc/images/chicago_paste.png" height="150" hspace="5"><img src="doc/images/photo_content.png" height="150" hspace="5"><img src="doc/images/content.gif" height="150" hspace="5">

<img src="doc/images/chicago_27.png" height="150" hspace="5"><img src="doc/images/in5_result.png" height="150" hspace="5"><img src="doc/images/test.gif" height="150" hspace="5">

## Prerequisites
- [Pytorch](http://pytorch.org/)
- [torchvision](https://github.com/pytorch/vision)
- [imageio](https://pypi.python.org/pypi/imageio) for GIF generation
- [opencv](https://opencv.org/) for video generation

**All code tested on Ubuntu 16.04, pytorch 0.4.1, and opencv 3.4.2**

## Image Style Transfer
- Clone from github: `git clone https://github.com/sunshineatnoon/LinearStyleTransfer`
- Download pre-trained models from [google drive](https://drive.google.com/open?id=1naZcc-Uw9xuFyR3cSyUjEJL7aOa1MfO7).
- Uncompress to root folder :
```
cd LinearStyleTransfer
unzip models.zip
rm models.zip
```
- Artistic style transfer
```
python TestArtistic.py --vgg_dir models/vgg_r41.pth --decoder_dir models/dec_r41.pth --matrixPath models/r41.pth --layer r41
```
or conduct style transfer on relu_31 features
```
python TestArtistic.py --vgg_dir models/vgg_r31.pth --decoder_dir models/dec_r31.pth --matrixPath models/r31.pth --layer r31
```
- Photo-real style transfer
```
python TestPhotoReal.py --vgg_dir models/vgg_r31.pth --decoder_dir models/dec_r31.pth --matrixPath models/r31.pth --layer r31
```
Note: images with `_filtered.png` as postfix are images filtered by bilateral filter after style transfer.

- Video style transfer
```
python TestVideo.py --vgg_dir models/vgg_r31.pth --decoder_dir models/dec_r31.pth --matrix_dir models/r31.pth --layer r31
```
- Real-time video demo
```
python real-time-demo.py --vgg_dir models/vgg_r31.pth --decoder_dir models/dec_r31.pth --matrixPath models/r31.pth --layer r31
```

## Model Training
### Data Preparation
- MSCOCO
```
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
```
- WikiArt
  - Either manually download from [kaggle](https://www.kaggle.com/c/painter-by-numbers).
  - Or install [kaggle-cli](https://github.com/floydwch/kaggle-cli) and download by running:
  ```
  kg download -u <username> -p <password> -c painter-by-numbers -f train.zip
  ```

### Training
To train a model that transfers relu4_1 features, run:
```
python Train.py --vgg_dir models/vgg_r41.pth --decoder_dir models/dec_r41.pth --layer r41 --contentPath PATH_TO_MSCOCO --stylePath PATH_TO_WikiArt --outf OUTPUT_DIR
```
or train a model that transfers relu3_1 features:
```
python Train.py --vgg_dir models/vgg_r31.pth --decoder_dir models/dec_r31.pth --layer r31 --contentPath PATH_TO_MSCOCO --stylePath PATH_TO_WikiArt --outf OUTPUT_DIR
```
Key hyper-parameters:
- style_layers: which features to compute style loss.
- style_weight: larger style weight leads to heavier style in transferred images.

Intermediate results and weight will be stored in `OUTPUT_DIR`

### Citation
```
@article{li2018learning,
  title={Learning Linear Transformations for Fast Arbitrary Style Transfer},
  author={Xueting Li and Sifei Liu and Jan Kautz and Ming-Hsuan Yang},
  journal={arXiv preprint arXiv:1808.04537},
  year={2018}
}
```
