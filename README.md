# capsnet
build capsnet

[original paper](https://arxiv.org/abs/1710.09829)

## Usage
Now,this program can only build capsnet.

In default,the model is built for MNIST.

```python
from capsnet import CapsNet,margin_loss
net = CapsNet()
criterion = margin_loss
'''
train block
'''
```
## ToDo
+ Implement decoder and reconstract-loss
+ fill docstrings and comment
+ train func, test func...etc


## Reference
Japanese(capsnetについての実装や説明をまとめたissue)
+ https://github.com/arXivTimes/arXivTimes/issues/488
Article
+ https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/
### Github
pytorch
+ https://github.com/XifengGuo/CapsNet-Pytorch
+ https://github.com/motokimura/capsnet_pytorch

tensorflow(original)
+ https://github.com/Sarasra/models/tree/master/research/capsules
