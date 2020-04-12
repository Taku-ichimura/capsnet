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
