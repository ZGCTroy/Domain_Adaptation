# BaseLine
BaseLine

# 7月17日 周三
Baseline BaselineMtoU， BaselineUtoM, BaselineStoM

* MNIST train [60000,1,28,28] test [10000,1,28,28]
* USPS train [7291,1,16,16] test [2007,1,16,16]
* SVHN train [73257,3,32,32] test [26032,3,32,32]

Baseline Resnet50 for Office-31

* Amazon 2817
* Dslr 498
* Webcam 795

Baseline Resnet50 for Office-Home

* Art 2427
* Clipart 4365
* Product 4439
* RealWorld 4357

# 7月18日
Add parser

first run on server


# 7月19日
完成Solver类的设计，DigitsBaselineSolver OfficeBaselineSolver
使用baseline在所有数据集上跑完

DANN 参考https://github.com/CuthbertCai/pytorch_DANN/blob/master

# 7月20日
重构代码，使baseline达到论文效果

# 7月21日
成功重构代码，更正optimizer错误，使baseline达到了论文效果

resnet = pretrained
Adam -> SGD

# 7月22日
完成DANN，设计base mode  + ad model

# 7月28日
MADA n_classes个discriminator的loss直接求和，不要求平均，使性能平均提升10%以上 
# 7月29
MCD 将一层的分类器拓展到多层，有助于提高效果,AtoW从84%提升到了89%


# 7月30日
init_weight

# 7月31日 

normalization 

image size resize=256 crop=224

MCD 的classifier的dropout很重要

# 8月1日

MADA 的 ad网络中不能加batchnormalize


