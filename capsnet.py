# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import transforms

def squash(s,dim=2):
    '''
    ベクトルの長さを０か１に近づける

    Parameters
    ----------
    s : Tensor
        入力カプセル
    dim : int 
        どの次元のnormを取るか
    Returns
    -------
    v : Tensor
        活性化されたベクトル
    '''
    norm = torch.norm(s, dim=2, keepdim=True)
    norm_sq = norm**2 
    tmp = s/ (1 + norm_sq)
    return tmp*norm

class PrimaryCaps(nn.Module):
    '''
    特徴マップをカプセルに変換するレイヤー

    処理は以下の通り

    1.畳み込まれた特徴マップを入力
        shape : [B,in_channel, height, width]
    2.もう一度畳み込む
        shape : [B,in_channel, [height-k]/stride+1, [width-k]/stride+1]
    3.特徴マップを8次元のカプセルに変形
        shape : [B,-1, in_capsuel_dim]
    4.squashで活性化
        shape : [B,-1, in_capsuel_dim]

    Attributes
    ----------
    in_channels : int
        入力された特徴マップの数
    stride : int 
        畳み込みのstride
    kernel : int 
        畳み込みのカーネルサイズ(kernel*kernel)
    in_capsule_dim : int
        カプセルの次元
    caps_unit : int
        カプセルのユニット数
    '''
    def __init__(self,in_channels=256,stride=2,kernel=9,in_capsuel_dim=8,caps_unit=32):
        super(PrimaryCaps, self).__init__()
        self.in_capsuel_dim = in_capsuel_dim
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=in_capsuel_dim*caps_unit,kernel_size=kernel,stride=stride)
        #He-Normalize
        nn.init.kaiming_normal_(self.conv.weight)
        
    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1,self.in_capsuel_dim)
        return squash(x,dim=-1)

class DigitCaps(nn.Module):
    '''
    カプセルを重み付きで全結合し、クラスカプセルに変換する

    処理は以下の通り

    1.PrimaryCapsで作ったカプセルを入力
        shape : [B, in_capsuel_num, in_capsuel_dim]
    2.上のカプセルを重ねてクラス数＊クラスカプセルの次元の数だけ増やす
        shape : [B, in_capsuel_num, in_capsuel_dim, num_classes*out_capsuel_dim]
    3.カプセルu_iをからクラスjに向かうカプセルを行列W_{ij}でaffine変換し、u_{j|i}の予測ベクトルを作る
        shape : [B, in_capsuel_num, in_capsuel_dim, num_classes, out_capsuel_dim]
    4.u_{j|i}をクラスカプセルv_jへ重みc_{ij}で足し込む、この重みはdynamic-routingで決める
        shape : [B, in_capsuel_num, num_classes]
    5.上でできたカプセルをsquashで活性化
        shape : [B, in_capsuel_num, num_classes]

    dynamic-routing
    1.ある8次元のカプセルu_iをあるクラスv_jに向かうカプセルに変換する
    　8×16行列W_{ij}でaffine変換し、16次元へ変換する。このとき、あるクラスを表す特徴カプセル同士は同じような方向へ向かう
    2.上の変換をすべてのカプセルに行い、重みc_{ij}で足してあるクラスを表すカプセルへ変換する
    　このc_{ij}はroutingによって決定され、あるカプセルがどのクラスカプセルに属するかを強調される用に決まる
    3.決められた重みでカプセルを足しこんでクラスカプセルを作る。このカプセルをsquashで活性化しクラスカプセルとする

    Attributes
    ----------
    num_classes : int
        分類するクラスの数
    num_capsuel : int
        PrimeCapsから来るカプセルの数
    out_capsuel_num : int
        クラスカプセルの次元
    in_capsuel_num : int
        PrimeCapsから来るカプセルの次元
    num_routing : int 
        routingを行う回数
    routing_bias : float default 0.1
        routingに使うバイアス
    '''
    def __init__(self,num_classes,num_capsuel,out_capsuel_dim,in_capsuel_dim,num_routing,routing_bias=0.1):
        super(DigitCaps,self).__init__()
        self.num_routing = num_routing
        self.num_classes = num_classes
        self.out_capsuel_dim = out_capsuel_dim
        self.routing_bias = routing_bias
        #[1152, 8, 10*16]
        self.W = nn.Parameter(torch.Tensor(num_capsuel,in_capsuel_dim,num_classes*out_capsuel_dim))
        nn.init.kaiming_normal_(self.W)
    def forward(self,x):
        '''
        ToDo:
            Tensorの変換に無駄が多いので軽くする
        '''
        batch_num,input_capsuel_num,_ = x.shape
        #[B, 1152,8]→[B, 1152, 8, 10*16]
        x = torch.stack([x]* self.num_classes*self.out_capsuel_dim,dim=3)
        #[B, 1152, 8, 10*16]→[B,1152,10*16]
        u_hat = torch.sum(x*self.W,dim=2)
        #[B,1152,10*16]→[B, 1152,10,16]
        u_hat = u_hat.view(batch_num,-1,self.num_classes,self.out_capsuel_dim)
        #勾配の計算に含まれないようにdynamic-routing用にコピー
        u_hat_detach = u_hat.detach()
        #[B, 1152,10,16]→[16, B, 1152, 10]
        u_hat_trans = u_hat.permute(3, 0, 1, 2)
        u_hat_detach_trans = u_hat_trans.detach()
        
        #[B, 10, 16]
        routing_biases = Variable(self.routing_bias*torch.ones(batch_num,self.num_classes,self.out_capsuel_dim)).to("cuda")
        #[B, 1152, 10]
        b = Variable(torch.zeros(batch_num,input_capsuel_num,self.num_classes)).to("cuda")
        for i in range(self.num_routing):
            #[B,1152,10]→[B,1152,10]
            c = F.softmax(b, dim=2)#sum over class dimension
            if i==self.num_routing -1:
                #[16, B, 1152, 10]→[16, B, 1152, 10]
                s = c*u_hat_trans
                #[16, B, 1152, 10]→[B, 1152, 10, 16]
                s = s.permute(1,2,3,0)
                #[B, 1152, 10, 16]→[B, 10, 16]
                s = torch.sum(s, axis=1) + routing_biases
                #[B, 10, 16]→[B, 10, 16]
                v = squash(s)
            else:
                #[16, B, 1152, 10]→[16, B, 1152, 10]
                s = c*u_hat_detach_trans
                #[16, B, 1152, 10]→[B, 1152, 10, 16]
                s = s.permute(1,2,3,0)
                #[B, 1152, 10, 16]→[B, 10, 16]
                s = torch.sum(s, dim=1) + routing_biases
                #[B, 10, 16]→[B, 10, 16]
                v = squash(s)
                 #[B, 10, 16]→[B, 1152, 10, 16]
                s = torch.stack([s]*input_capsuel_num,dim=1)
                #[B, 1152, 10, 16]→[B, 1152, 10]
                distances = torch.sum(u_hat_detach*s,dim=3)
                b += distances
        return v
    

class CapsNet(nn.Module):
    '''
    CapsuelNetを構築するクラス
    
    処理の流れはMNIST画像(1,28,28)を例に取ると
    
    1.画像入力
        input_image : [batch , 1, 28, 28]
    2.畳み込み＋RELU活性化で特徴マップ作成
        conved_image : [batch, 256, 20, 20]
    3.primecapsで特徴マップをベクトルで切り出し、squashで活性化したカプセルに変形
        prime_caps : [batch, 256, 6, 6]
    4.8次元のベクトルに変形
        prime_caps : [batch, 1152, 8]
    5.Digitcapsでカプセルをaffin変換し、16次元のクラスカプセルへ重み付きで変換し活性化
        digit_caps : [batch, 10, 16]
    (6.クラスのカプセルからdecoderを通して元の画像を再構成)
        recon_image : [batch, 1, 28, 28]
    

    Attributes
    ----------
    in_capsuel_dim : int
        PrimeCapsのベクトルの次元
    out_capsuel_dim : int
        クラスカプセルの次元
    num_classes : int
        分類するクラスの数
    num_routing : int
        dynamic-routingを行う数
    conv : layer
        最初の畳み込み層
    relu : layer
        Relu活性化
    prime : layer
        PrimeCaps layer
    digit : layer
        DigitCaps layer
    decoder : layer
        Decoder layer

    '''
    def __init__(self,in_capsuel_dim=8,out_capsuel_dim=16,num_classes=10,num_routing=3,use_decoder=True):
        super(CapsNet,self).__init__()
        self.in_capsuel_dim = in_capsuel_dim
        self.out_capsuel_dim = out_capsuel_dim
        self.num_classes = num_classes
        self.use_decoder = use_decoder
        
        self.conv = nn.Conv2d(in_channels = 1,out_channels=256,kernel_size=9,stride=1)
        self.relu = nn.ReLU()
        #He-Normalize
        nn.init.kaiming_normal_(self.conv.weight)
        self.prime = PrimaryCaps()
        self.digit = DigitCaps(num_classes,1152,out_capsuel_dim,in_capsuel_dim,num_routing)

        self.decoder = nn.Sequential(
            nn.Linear(16*self.num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 28*28),
            nn.Sigmoid()
        )
    
    def forward(self,x,y=None):
        #[B,1,28,28]→[B,256,20,20]
        x = self.relu(self.conv(x))
        #[B,256,20,20]→[B,1152,8]
        x = self.prime(x)
        #[B,1152,8]→[B,10,16]
        x = self.digit(x)
        #[B,10,16]→[B,10]
        out = x.norm(dim=-1)
        if self.use_decoder:
            if y is None:
                index = out.max(dim=1)[1]
                y = Variable(torch.zeros(out.size()).scatter_(1, index.view(-1, 1).to("cpu").data, 1.).to("cuda"))
            recon = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
            recon = recon.view(-1,1,28,28)
        else:
            recon=None
        return out,recon

def margin_loss(labels, output, margin=0.4, downweight=0.5):
    '''
    margin-lossを計算
    ターゲットクラスの確率が0.9に近く、その他クラスの確率が0.1に近ければ得する関数
    つまり、ターゲットクラスのカプセルのノルムを強調し、
    それ以外のカプセルのノルムを潰す方向に学習が進む

    論文と関数になっているが、λの値が0.1→0.025になっている

    Parameters
    ----------
    labels : Tensor
        正解ラベルのone-hot vector
        shape : [batch , num_classes]
    output : Tensor
        モデルが出力したそのクラスの確率
        shape : [batch , num_classes]
    margin : float
        0.5+marginが目的のターゲットクラスの目標ノルムとなる
    downweight : float
        ターゲットクラス外由来のlossの重み

    Returns
    -------
    loss : float
        計算されたmargin-lossの値
    '''
    logits = output - 0.5
    margin = margin*torch.ones(logits.shape).to("cuda")
    #max(0,margin+0.5-logit)^2
    positive_cost = labels * torch.gt(-logits, -margin) * torch.pow(logits - margin, 2)
    #max(0,margin-0.5-logit)^2
    negative_cost = (1 - labels) * torch.gt(logits, -margin) * torch.pow(logits + margin, 2)
    L_margin = 0.5 * positive_cost + downweight * 0.5 * negative_cost
    L_margin = L_margin.sum(dim=1).mean()
    return L_margin

def reconstruct_loss(x,recon,alpha):
    '''
    reconstruction-lossを計算
    再構成された画像とのMSE

    Parameters
    ----------
    x : Tensor
        入力画像
        shape : [batch , in_channel , height , width]
    recon : Tensor
        outputのcapsuelから再構成された画像
        shape : [batch , in_channel , height , width]
    alpha : float
        reconstruction-lossの重み、論文値は0.0005

    Returns
    -------
    loss : float
        計算されたreconstruntion-lossの値
    '''
    loss = nn.MSELoss()(x,recon)
    return loss*alpha

def total_loss(labels, output,x,recon,alpha=0.0005):
    '''
    margin-loss + reconstruction-lossを計算
    Parameters
    ----------
    labels : Tensor
        正解ラベルのone-hot vector
        shape : [batch , num_classes]
    output : Tensor
        モデルが出力したそのクラスの確率
        shape : [batch , num_classes]
    x : Tensor
        入力画像
        shape : [batch , in_channel , height , width]
    recon : Tensor
        outputのcapsuelから再構成された画像
        shape : [batch , in_channel , height , width]
    alpha : float
        reconstruction-lossの重み、論文値は0.0005

    Returns
    -------
    loss : float
        計算されたロスの値(margin-loss + reconstruction-loss)
    '''
    L_margin = margin_loss(labels, output)
    L_recon = reconstruct_loss(x,recon,alpha)
    return L_margin + L_recon

def train(net,criterion,optimizer,sheduler,trainloader,epochs=50):
    '''
    モデルを訓練する関数
    epochごとの重みをcaps_weightに保存する
    caps_weightがなければ最初に作成する

    Parameters
    ----------
    net : model
        モデル
    criterion : function
        使用する損失関数
    optimizer : optim
        使用する勾配法
    sheduler : optim.lr_scheduler
        学習率のスケジューラー
    trainloader : dataloader
        訓練データのデータローダー
    epochs : int default 50
        エポック数
    '''
    os.makedirs("./caps_weight", exist_ok=True)
    for epoch in range(epochs):
        train_loss = 0
        correct = 0
        sheduler.step()
        for i,(image,label) in enumerate(tqdm(trainloader)):
            label = torch.zeros(label.size(0), 10).scatter_(1, label.view(-1, 1), 1.)
            optimizer.zero_grad()
            image = image.to("cuda")
            label = label.to("cuda")
            out,recon = net(image,label)
            loss = criterion(labels=label, output=out,recon=recon,x = image)
            loss.backward()
            optimizer.step()
            out = out.max(1)[1]
            label = label.data.max(1)[1]
            correct+= out.eq(label).cpu().sum()
            train_loss += loss.item()

            print("\r {}/{}-loss:{},acc:{}".format(epoch,50,train_loss/(i+1),1.0*correct/(i+1)/100), end="")
        print("epoch:{}".format(epoch))
        print(train_loss/len(trainloader))
        print(correct.item() / len(trainloader.dataset))
        torch.save(net.state_dict(), "./caps_weight/epoch_"+str(epoch)+"_capsnet_weight.pth")

def test(net,criterion,testloader):
    '''
    モデルの性能をテストする関数

    Parameters
    ----------
    net : model
        モデル
    criterion : function
        使用する損失関数
    testloader : dataloader
        テストデータのデータローダー
    '''
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)
            x = x.to("cuda")
            y = y.to("cuda")
            y_pred,recon = net(x)
            test_loss += criterion(labels=y, output=y_pred,recon=recon,x = x)
            y_pred = y_pred.max(1)[1]
            y_true = y.data.max(1)[1]
            correct += y_pred.eq(y_true).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print("epoch:{} acc:{} loss:{}".format(epoch,test_ac,test_loss.item()))
        
if __name__ == "__main__":
    #set parameter
    epochs = 50
    batch_size = 100
    initial_lr = 0.0001
    decay_rate = 0.9

    net = CapsNet()
    net.to("cuda")

    transform = transforms.Compose(
    [transforms.RandomCrop(size=28, padding=2),
        transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(root='./datasets/MNIST', 
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=2)

    testset = torchvision.datasets.MNIST(root='./datasets/MNIST', 
                                            train=False, 
                                            download=True, 
                                            transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                                batch_size=10,
                                                shuffle=False, 
                                                num_workers=2)

    criterion = total_loss#caps_loss
    optimizer = optim.Adam(net.parameters(),lr=initial_lr)
    sheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

    #train
    train(net,criterion,optimizer,sheduler,trainloader,epochs=epochs)
    #test
    for epoch in range(50):
        name = "./caps_weight/epoch_"+str(epoch)+"_capsnet_weight.pth"
        net.load_state_dict(torch.load(name))
        net.to("cuda")
        test_loss,test_ac = test(net,testloader)

