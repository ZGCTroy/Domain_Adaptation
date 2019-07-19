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