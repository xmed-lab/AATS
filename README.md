# Leveraging Anatomical Consistency for Multi-Object Detection in Ultrasound Images via Source-free Unsupervised Domain Adaptation

<a src="https://img.shields.io/badge/%F0%9F%9A%80-xmed_Lab-ed6c00.svg?style=flat-square" href="https://xmengli.github.io/">
<img src="https://img.shields.io/badge/%F0%9F%9A%80-xmed_Lab-ed6c00.svg?style=flat-square">
</a>

## PostScript
 This project is the pytorch implemention of AATS;

 Our experimental platform is configured with <u>One *RTX3090 (cuda>=11.0)*</u>; 

 Currently, this code is avaliable for public dataset <a href="https://github.com/xmed-lab/GraphEcho">CardiacUDA</a> and <a href="https://github.com/xmed-lab/ToMo-UDA">FUSH</a>;  

## Installation

### Prerequisites

- Python ≥ 3.6
- PyTorch ≥ 1.5 and torchvision that matches the PyTorch installation.
- Detectron2 == 0.5

### Install python env

To install required dependencies on the virtual environment of the python (e.g., virtualenv for python3), please run the following command at the root of this code:
```
$ python3 -m venv /path/to/new/virtual/environment/.
$ source /path/to/new/virtual/environment/bin/activate
```
For example:
```
$ mkdir python_env
$ python3 -m venv python_env/
$ source python_env/bin/activate
```
 

### Build Detectron2 from Source

Follow the [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install Detectron2.

## Dataset download

1. Download the datasets


2. Organize the dataset as the COCO annotation format.

## Training

- Train the AATS under Center 1 of Heart (source) and Center 2 of Heart (target) on FUSH dataset

```shell
python train_net.py \
      --num-gpus 1 \
      --config configs/sfda_at_rcnn_vgg_fetus_4c_1to2.yaml\
      OUTPUT_DIR output/AATS_4c_1to2
```

- Train the AATS under Center 2 of Heart (source) and Center 1 of Heart (target) on FUSH dataset

```shell
python train_net.py\
      --num-gpus 1\
      --config configs/sfda_at_rcnn_vgg_fetus_4c_2to1.yaml\
      OUTPUT_DIR output/AATS_4c_2to1
```

## Resume the training

```shell
python train_net.py \
      --resume \
      --num-gpus 1 \
      --config configs/sfda_at_rcnn_vgg_fetus_4c_1to2.yaml MODEL.WEIGHTS <your weight>.pth
```

## Evaluation

```shell
python train_net.py \
      --eval-only \
      --num-gpus 1 \
      --config configs/sfda_test.yaml \
      MODEL.WEIGHTS <your weight>.pth
```

## Results and Model Weights

We will publish the VGG pre-training weights and model weights soon.

<!-- ## Citation

If you use Adaptive Teacher in your research or wish to refer to the results published in the paper, please use the following BibTeX entry.

```BibTeX
@inproceedings{li2022cross,
    title={Cross-Domain Adaptive Teacher for Object Detection},
    author={Li, Yu-Jhe and Dai, Xiaoliang and Ma, Chih-Yao and Liu, Yen-Cheng and Chen, Kan and Wu, Bichen and He, Zijian and Kitani, Kris and Vajda, Peter},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
} 
``` -->
## Code Reference 
  - [detectron2](https://github.com/facebookresearch/detectron2)
  - [Adaptive Teacher](https://yujheli.github.io/projects/adaptiveteacher.html)

