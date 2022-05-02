import os
import torch.backends.cudnn as cudnn
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision.datasets as datasets
from data_loader import GetData

cuda = True
cudnn.benchmark = True


def test(dataset_name):
    assert dataset_name in ['MNIST', 'mnist_m']

    model_root = 'models'
    image_root = os.path.join('dataset', dataset_name)
    batch_size = 128
    alpha = 0

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

    if dataset_name == 'mnist_m':
        data_test_list = os.path.join(image_root, 'mnist_m_test_labels.txt')
        dataset = GetData(data_root=os.path.join(image_root, 'mnist_m_test'),
                          data_list=data_test_list, transform=img_target_trans)
    else:
        dataset = datasets.MNIST(root='dataset', transform=img_source_trans, train=False)

    dataset_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    test_model = torch.load(os.path.join(model_root, 'mnist_mnistm_model_epoch_current.pth'))
    test_model = test_model.eval()

    if cuda:
        test_model = test_model.cuda()

    len_dataloader = len(dataset_loader)
    dataset_iter = iter(dataset_loader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        d_img, d_label = dataset_iter.next()

        batch_size = len(d_label)

        if cuda:
            d_img = d_img.cuda()
            d_label = d_label.cuda()

        class_output, _ = test_model(input_data=d_img, alpha=alpha)
        """
        torch.max(input, dim, max=None, max_indices=None) -> (Tensor, LongTensor)
        返回输入张量给定维度上的最大值，并同时返回每个最大值的位置索引（两个数组）
        [1]返回的是个最大值的位置索引的数组
        dim (int) – 指定的维度 0列 1行
        """
        prediction = class_output.data.max(1, keepdim=True)[1]
        n_correct += prediction.eq(d_label.view_as(prediction)).cpu().sum()
        n_total += batch_size

        i += 1
    acc = n_correct.data.numpy() * 1.0 / n_total

    return acc
