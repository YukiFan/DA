import torch.utils.data as data
from PIL import Image
import os


class GetData(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        # 类似与C语言传过来的参数被赋值给class类本身的成员(self.something)
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []
        for line in data_list:
            # 这里data是一个一个的字符串00000000.png 5 \n
            # 所以获得[:-3]代表从低0列开始直到-1（最后一列），-2，-3（倒数第3列）
            # data[-2] 对于一维的数据类似数组可以直接取data[0]，data[1]，所以这里取倒数第2列
            # print(data[0])
            # print(data[1])
            # print(data[2])
            # print(data[3])
            self.img_paths.append(line[:-3])
            self.img_labels.append(line[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data
