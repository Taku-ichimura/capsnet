# capsnet
build capsnet

[original paper](https://arxiv.org/abs/1710.09829)

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
## ToDo
+ train func, test func...etc


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
