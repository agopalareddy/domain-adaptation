# Domain Adaptation With Domain-Adversarial Training of Neural Networks

 <div align="center">
    <a href="https://colab.research.google.com/github/reshalfahsi/domain-adaptation/blob/master/Domain_Adaptation_With_Domain_Adversarial_Training_of_Neural_Networks.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a>
    <br />
 </div>


Domain adaptation's main objective is to adapt the model trained on the source dataset in which the label is available to perform decently on the target dataset, which has a pertinent distribution yet the label is not already on hand. In this project, the pretrained RegNetY_400MF is leveraged as the model undergoing the adaptation procedure. The procedure is conducted with Domain-Adversarial Training of Neural Networks or DANN. Succinctly, DANN works by adversarially training the appointed model on the source dataset along with the target dataset. DANN uses an extra network as the domain classifier (the critic or discriminator) and applies a gradient reversal layer to the output of the feature extractor. Thus, the losses accounted for this scheme are the classification head loss (the source dataset) and the domain loss (the source dataset and the target dataset). Here, the source dataset is MNIST and the target dataset is SVHN. On MNIST, various data augmentations (geometric and photometric) are utilized on the fly during training. To monitor the adaptation performance, the testing set of SVHN is designated as the validation and testing set.


## Experiment

Study the adaptation process by following [the link to the notebook](https://github.com/reshalfahsi/domain-adaptation/blob/master/Domain_Adaptation_With_Domain_Adversarial_Training_of_Neural_Networks.ipynb) quenching your curiosity.


## Result

## Quantitative Result

The model's performance on the target dataset:

Test Metric  | Score
------------ | -------------
Loss         | 3.138
Accuracy     | 44.79%


## Accuracy and Loss Curve

<p align="center"> <img src="https://github.com/reshalfahsi/domain-adaptation/blob/master/assets/acc_curve.png" alt="acc_curve" > <br /> Accuracy curves of the model on the source dataset (MNIST) and the target dataset (SVHN). </p>

<p align="center"> <img src="https://github.com/reshalfahsi/domain-adaptation/blob/master/assets/loss_curve.png" alt="loss_curve" > <br /> Loss curves of the model on the source dataset (MNIST) and the target dataset (SVHN). </p>


## Qualitative Result

The collated image below visually reports the prediction results on the target dataset.

<p align="center"> <img src="https://github.com/reshalfahsi/domain-adaptation/blob/master/assets/qualitative.png" alt="qualitative" > <br /> Some results on the SVHN dataset. </p>


## Credit

- [Domain-Adversarial Training of Neural Networks](https://arxiv.org/pdf/1505.07818.pdf)
- [Unsupervised Domain Adaptation by Backpropagation](http://sites.skoltech.ru/compvision/projects/grl/files/paper.pdf)
- [DANN](https://github.com/fungtion/DANN)
- [Designing Network Design Spaces](https://arxiv.org/pdf/2003.13678.pdf)
- [Reading Digits in Natural Images with Unsupervised Feature Learning](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf)
- [TorchVision's SVHN](https://github.com/pytorch/vision/blob/main/torchvision/datasets/svhn.py)
- [Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791)
- [TorchVision's MNIST](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py)
- [Semi-supervision and domain adaptation with AdaMatch](https://keras.io/examples/vision/adamatch/)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
