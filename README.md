# CapsNet
build capsnet

[original paper](https://arxiv.org/abs/1710.09829)

## Requirements
+ tqdm
+ numpy
+ torch
+ torchvision

## Usage
Now,this program can build capsnet and train.

In default,the model is built for MNIST.

```python
from capsnet import CapsNet,total_loss
net = CapsNet()
criterion = total_loss
'''
train block
'''
```
## result

![Screenshot from 2020-04-19 13-26-10](https://user-images.githubusercontent.com/22934822/79679494-7378c080-8241-11ea-8771-07f75605ba71.png)

Achieved 99.61% of MNIST's test data

settings is same as main func

## Reference(code & theory)
capsnetについての実装や説明をまとめたissue
+ https://github.com/arXivTimes/arXivTimes/issues/488

Article
+ https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/
### Github
pytorch
+ https://github.com/XifengGuo/CapsNet-Pytorch
+ https://github.com/motokimura/capsnet_pytorch

original program(Tensorflow)
+ https://github.com/Sarasra/models/tree/master/research/capsules
