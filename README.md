# SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation (NeurIPS 2022)

![](resources/flops.png)

The repository contains official Pytorch implementations of training and evaluation codes and pre-trained models for **SegNext**. 

The paper is in [Here](https://arxiv.org/pdf/2209.08575.pdf).

Our implementation is mainly based on (https://github.com/visual-attention-network/segnext). And the original code is based on [MMSegmentaion v0.24.1](https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1).


## Citation
If you find our repo useful for your research, please consider citing our paper:

```
@article{guo2022segnext,
  title={SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation},
  author={Guo, Meng-Hao and Lu, Cheng-Ze and Hou, Qibin and Liu, Zhengning and Cheng, Ming-Ming and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2209.08575},
  year={2022}
}

```

## Results

**Notes**: ImageNet Pre-trained models can be found in [TsingHua Cloud](https://cloud.tsinghua.edu.cn/d/c15b25a6745946618462/).


## Installation
Install the dependencies and download ADE20K according to the guidelines in [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/v0.24.1/docs/en/get_started.md#installation).


```
pip install timm
cd SegNeXt
python setup.py develop
```

## Training

Original SegNeXt used 8 GPUs for training by default. However, we changed it to 1 GPU because of limited computational resources. Run:

For Landcover.ai:
```bash
python tools/train.py /work/yl407/SegNeXt/local_configs/segnext/large/segnext.large.512x512.landcover.40k_weight.py
python tools/train.py /work/yl407/SegNeXt/local_configs/segnext/large/segnext.large.512x512.landcover.40k_weight_sub.py
python tools/train.py /work/yl407/SegNeXt/local_configs/segnext/large/segnext.large.512x512.landcover.40k_weight_sub_nopre.py
```
For Pascal VOC:
```bash
python tools/train.py /work/yl407/SegNeXt/local_configs/segnext/large/segnext.large.512x512.voc.40k.py
python tools/train.py /work/yl407/SegNeXt/local_configs/segnext/large/segnext.large.512x512.voc.40k_sub.py
python tools/train.py /work/yl407/SegNeXt/local_configs/segnext/large/segnext.large.512x512.voc.40k_sub_nopre.py
```


## Evaluation

To evaluate the model, run:

For Landcover.ai:
```bash
python tools/test.py  /work/yl407/SegNeXt/local_configs/segnext/large/segnext.large.512x512.voc.40k.py /work/yl407/SegNeXt/work_dirs/segnext.large.512x512.voc.40k/latest.pth --eval mIoU
python tools/test.py  /work/yl407/SegNeXt/local_configs/segnext/large/segnext.large.512x512.voc.40k_sub.py /work/yl407/SegNeXt/work_dirs/segnext.large.512x512.voc.40k_sub/latest.pth --eval mIoU
python tools/test.py  /work/yl407/SegNeXt/local_configs/segnext/large/segnext.large.512x512.voc.40k_sub_nopre.py /work/yl407/SegNeXt/work_dirs/segnext.large.512x512.voc.40k_sub_nopre/latest.pth --eval mIoU
```


For Pascal VOC:
```bash
python tools/test.py  /work/yl407/SegNeXt/local_configs/segnext/large/segnext.large.512x512.landcover.40k_weight.py /work/yl407/SegNeXt/work_dirs/segnext.large.512x512.landcover.40k_weight/latest.pth --eval mIoU
python tools/test.py  /work/yl407/SegNeXt/local_configs/segnext/large/segnext.large.512x512.landcover.40k_weight_sub.py /work/yl407/SegNeXt/work_dirs/segnext.large.512x512.landcover.40k_weight_sub/latest.pth --eval mIoU
python tools/test.py  /work/yl407/SegNeXt/local_configs/segnext/large/segnext.large.512x512.landcover.40k_weight_sub_nopre.py /work/yl407/SegNeXt/work_dirs/segnext.large.512x512.landcover.40k_weight_sub_nopre/latest.pth --eval mIoU
```

## Changes we made to the original SegNext implementation:

* Change the multi-processing setup because of lack of computational resources.
* To run Pascal VOC smoothly, we need to create customized data processing files and config files for it because SegNext didn't provide it.
* To run our customized remote sensing dataset,  we need to create LandcoverDataset class,  register the customized dataset, create script to convert dataset, create dataset processing file, and config files.
\end{itemize}

## Where to find each model in the framework:

* Data processing (customized): SegNeXt/local\_configs/\_base\_/datasets/. In this directory, I created separate processing file for each dataset.
* Encoder: Model is in SegNeXt/mmseg/models/backbones/mscan.py, including MLP (line 15-35), StemConv (line 38-56), Attention (line 59-91), Spatial Attention (line 94-110), Image to Patch Embedding (line 148-167), organizing MSCAN class (line 171-255), Depth-wise convolution and Scale (line 258-266) 

   Parameter set up is in ~/SegNeXt/local\_configs/\_base\_/models/
* Decoder: Model is in SegNeXt/mmseg/models/decode\_heads/ham\_head.py, including Matrix Decomposition (line 11-100), NMF by calling Matrix Decomposition (line 103-145), Hamburger class by calling NMF (line 148-180), and LightHamHead class by calling Hamburger (line 183-240)

    Parameter set up is in ~/SegNeXt/local\_configs/\_base\_/models/



## Acknowledgment

Our implementation is mainly based on (https://github.com/visual-attention-network/segnext)

SegNeXt implementation is mainly based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1), [Segformer](https://github.com/NVlabs/SegFormer) and [Enjoy-Hamburger](https://github.com/Gsunshine/Enjoy-Hamburger). Thanks for their authors.

## LICENSE

This repo is under the Apache-2.0 license. For commercial use, please contact the authors.
