## Reproducing MT, DANN, MCD, MADA in pytorch
* Baseline 
    * [DigitsMU](https://arxiv.org/abs/1706.05208)  
    * [DigitsStoM](https://arxiv.org/abs/1706.05208)
    * [ResNet50](http://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

    [Network model](https://github.com/ZGCTroy/Domain_Adaptation/tree/master/networks/Baseline.py)
    
    [Baseline solver](https://github.com/ZGCTroy/Domain_Adaptation/tree/master/solvers/BaselineSolver.py)
    
* [DANN]

    [Paper](http://www.jmlr.org/papers/volume17/15-239/15-239.pdf)
    
    [Network model](https://github.com/ZGCTroy/Domain_Adaptation/tree/master/networks/DANN.py)
    
    [DANN solver](https://github.com/ZGCTroy/Domain_Adaptation/tree/master/solvers/DANNSolver.py)

    Ganin Y, Ustinova E, Ajakan H, et al. Domain-adversarial training of neural networks[J]. The Journal of Machine Learning Research, 2016, 17(1): 2096-2030.

* [MT]

    [Paper](https://arxiv.org/abs/1706.05208)

    [Network model](https://github.com/ZGCTroy/Domain_Adaptation/tree/master/networks/MT.py)
    
    [MT solver](https://github.com/ZGCTroy/Domain_Adaptation/tree/master/solvers/MTSolver.py)
    
    French G, Mackiewicz M, Fisher M. Self-ensembling for visual domain adaptation[J]. arXiv preprint arXiv:1706.05208, 2017.

* [MCD]

    [Paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.html)

    [Network model](https://github.com/ZGCTroy/Domain_Adaptation/tree/master/networks/MCD.py)
    
    [MCD solver](https://github.com/ZGCTroy/Domain_Adaptation/tree/master/solvers/MCDSolver.py)
    
    Saito K, Watanabe K, Ushiku Y, et al. Maximum classifier discrepancy for unsupervised domain adaptation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 3723-3732.

* [MADA]

    [Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/17067)
    
    [Network model](https://github.com/ZGCTroy/Domain_Adaptation/tree/master/networks/MADA.py)
    
    [MADA solver](https://github.com/ZGCTroy/Domain_Adaptation/tree/master/solvers/MADASolver.py)
    
    Pei Z, Cao Z, Long M, et al. Multi-adversarial domain adaptation[C]//Thirty-Second AAAI Conference on Artificial Intelligence. 2018.