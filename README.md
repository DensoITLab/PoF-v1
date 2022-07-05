# PoF-v1
Code of PoF: Post-Training of Feature Extractor for Improving Generalization [1].  The source codes reproduce the results in [1] using Wide ResNet [2] trained on CIFAR-10, CIFAR-100[3], SVHN[4], and Fashion-MNIST [5], with data augmentation including cutout [6].

Trained parameters can be found here: https://d-itlab.s3.ap-northeast-1.amazonaws.com/PoF/pretrained_models.zip
WRN_C10_pretrained_bySAM: WideResNet-28-10 trained with SAM on CIFAR-10 for 200 epochs (same setting as in Table 1 in [1]).
WRN_C10_pretrained_bySAM2PoF: WideResNet-28-10 post-trained with PoF on CIFAR-10 for 50 epochs, started with WRN_C10_pretrained_bySAM (same setting as in Table 1 in [1]).

## Requirements

- computer running Linux
- NVIDIA GPU and NCCL
- Python version 3.7
- PyTorch

## Usage

First, use python train.py to create a pretrained model by a previous optimization method. 
Then, use python train.py again to post-train a pretrained model by PoF.
Here are some examples setting:
To create a pretrained model.  (default setting: --dataset='cifar10' --optimizer=='momentumSGD', --flgDist=false)
> $python3 train.py 

To post-train a pretrained model.  (PretrainedSave must be rewritten to the correct path)
> $python3 train.py --flgContinue=true --flgPoF=true --PretrainedSave='./result/save/pret'

## Author

Ryota Yamada, Tokyo Institute of Technology
Guoquing Liu, (formerly) Denso IT Laboratory, Inc.


## Reference

[1] Ikuro Sato, Ryota Yamada, Masayuki Tanaka, Nakamasa Inoue and Rei Kawakami, "PoF: Post-Training of Feature Extractor for Improving Generalization", Proceedings of the 39th International Conference on Machine Learning (ICML), 2022.

[2] Sergey Zagoruyko and Nikos Komodakis, "Wide Residual Networks", Proceedings of the British Machine Vision Conference (BMVC), 2016.

[3] Alex Krizhevskyf and Geoffrey Hinton, "Learning multiple layers of features from tiny images", Technical Report, University of Toronto, 2009.

[4] Netzer Y, Wang T, Coates A, Bissacco A, Wu B, and Ng A. Y, "Reading digits in natural images with unsupervised feature learning", NeurIPS Workshop, 2011.

[5] Xiao H, Rasul K, and Vollgraf R, "Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms", 2017

[6] DeVries, T., & Taylor, G. W. (2017). Improved regularization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552.

## LICENSE

Copyright (C) 2022 Denso IT Laboratory, Inc.
All Rights Reserved

Denso IT Laboratory, Inc. retains sole and exclusive ownership of all
intellectual property rights including copyrights and patents related to this
Software.

Permission is hereby granted, free of charge, to any person obtaining a copy
of the Software and accompanying documentation to use, copy, modify, merge,
publish, or distribute the Software or software derived from it for
non-commercial purposes, such as academic study, education and personal use,
subject to the following conditions:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
