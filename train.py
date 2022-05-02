import os
import sys
import random
import numpy as np
import torch.nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torch.utils.data import DataLoader
from data_loader import GetData
from model import CNNModel
import torch.optim as optim
from test import test

cuda = True
cudnn.benchmark = True
batch_size = 128
source_dataset_name = 'MNIST'
target_dataset_name = 'mnist_m'
source_image_root = os.path.join('dataset', source_dataset_name)
target_image_root = os.path.join('dataset', target_dataset_name)
lr = 1e-3
n_epoch = 100
model_root = 'models'

# 设置CPU生成随机数的种子，方便下次复现实验结果，在每次重新运行程序时，同样的随机数生成代码得到的是同样的结果
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

img_source_trans = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

img_target_trans = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_source_dataset = datasets.MNIST(root='dataset', download=True, transform=img_source_trans, train=True)
# num_worker：创建多线程，提前加载未来会用到的batch数据
train_source_loader = DataLoader(dataset=train_source_dataset, shuffle=True, batch_size=batch_size, num_workers=8)

train_target_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')

train_target_dataset = GetData(
    data_root=os.path.join(target_image_root, 'mnist_m_train'),
    data_list=train_target_list,
    transform=img_target_trans)
train_target_loader = DataLoader(dataset=train_target_dataset, shuffle=True, batch_size=batch_size, num_workers=8)

model = CNNModel()

optimizer = optim.Adam(model.parameters(), lr=lr)
# CrossEntropyLoss (x,y)=NLLLoss(logSoftmax(x),y)
loss_class = torch.nn.NLLLoss()
domain_loss = torch.nn.NLLLoss()

if cuda:
    model = model.cuda()
    loss_class = loss_class.cuda()
    domain_loss = domain_loss.cuda()

'''
根据PyTorch的自动求导机制
如果一个tensor设置require_grad为True的情况下
才会对这个tensor以及由这个tensor计算出来的其他tensor求导
并将导数值存在tensor的grad属性中
便于优化器来更新参数
所以一般情况下
只有对训练过程的tensor设置require_grad为True
便于求导和更新参数
而验证过程只是检查当前模型的泛化能力
只需要求出loss
所以不需要根据val set的loss对参数进行更新
因此require_grad为False
'''
for p in model.parameters():
    p.requires_grad = True

best_rate = 0.0
for epoch in range(n_epoch):
    len_dataloader = min(len(train_target_loader), len(train_source_loader))
    source_iter = iter(train_source_loader)
    target_iter = iter(train_target_loader)

    for i in range(len_dataloader):
        # 这里作者设置的是一个可变的lamda，避免早期过多的噪声影响
        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        data_source = source_iter.next()
        s_img, s_label = data_source
        # 训练的过程通常使用mini-batch方法，所以如果不将梯度清零的话，梯度会与上一个batch的数据相关，因此该函数要写在反向传播和梯度下降之前
        model.zero_grad()
        # batch_size = len(s_label)  ！！！！这里不断变换的原因是每一个batch size不一定是真的完全128的，可能有不平均的情况
        # 所以label以及相应的domain_label的大小设置也需要改变，不然会报错
        batch_size = len(s_label)
        # torch.long():向下取整，domain_label是0和1,0代表源域，1代表目标域
        # torch.zeros(batch_size)：返回一个形状为为batch_size,类型为torch.dtype，里面的每一个值都是0的tensor
        # 所以这里domain_label是全部设为0，对应源域
        source_domain = torch.zeros(batch_size).long()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            source_domain = source_domain.cuda()

        out_label, out_domain = model(input_data=s_img, alpha=alpha)
        loss_label = loss_class(out_label, s_label)
        loss_domain = domain_loss(out_domain, source_domain)

        data_target = target_iter.next()
        t_img, _ = data_target
        batch_size = len(t_img)
        # 所以这里domain_label是全部设为1，对应目标域
        target_domain = torch.ones(batch_size).long()

        if cuda:
            t_img = t_img.cuda()
            target_domain = target_domain.cuda()

        _, out_domain_t = model(input_data=t_img, alpha=alpha)
        loss_domain_t = domain_loss(out_domain_t, target_domain)

        LOSS = loss_label + loss_domain + loss_domain_t

        LOSS.backward()
        optimizer.step()

        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                         % (epoch, i + 1, len_dataloader, loss_label.data.cpu().numpy(),
                            loss_domain.data.cpu().numpy(), loss_domain_t.data.cpu().item()))
        sys.stdout.flush()
        torch.save(model, '{0}/mnist_mnistm_model_epoch_current.pth'.format(model_root))

    print('\n')
    accu_s = test(source_dataset_name)
    print('Accuracy of the %s dataset: %f' % ('mnist', accu_s))
    accu_t = test(target_dataset_name)
    print('Accuracy of the %s dataset: %f\n' % ('mnist_m', accu_t))
    if accu_t > best_rate:
        best_accu_s = accu_s
        best_rate = accu_t
        torch.save(CNNModel, '{0}/mnist_mnistm_model_epoch_best.pth'.format(model_root))

print('============ Summary ============= \n')
print('Accuracy of the %s dataset: %f' % ('mnist', best_accu_s))
print('Accuracy of the %s dataset: %f' % ('mnist_m', best_rate))
print('Corresponding model was save in ' + model_root + '/mnist_mnistm_model_epoch_best.pth')
