import torch.nn as nn
from grl import RevGrad


# super().__init__()的作用：执行父类的构造函数，使得我们能够调用父类的属性

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1)
        )

        self.domian_classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature_extractor(input_data)
        # view类似于reshape，一个参数定为-1，代表动态调整这个维度上的元素个数，以保证元素的总数保持不变
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = RevGrad.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domian_classifier(reverse_feature)

        return class_output, domain_output

# import torch.nn as nn
# from grl import RevGrad
#
#
# class CNNModel(nn.Module):
#
#     def __init__(self):
#         super(CNNModel, self).__init__()
#         self.feature = nn.Sequential()
#         self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
#         self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
#         self.feature.add_module('f_pool1', nn.MaxPool2d(2))
#         self.feature.add_module('f_relu1', nn.ReLU(True))
#         self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
#         self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
#         self.feature.add_module('f_drop1', nn.Dropout2d())
#         self.feature.add_module('f_pool2', nn.MaxPool2d(2))
#         self.feature.add_module('f_relu2', nn.ReLU(True))
#
#         self.class_classifier = nn.Sequential()
#         self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
#         self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
#         self.class_classifier.add_module('c_relu1', nn.ReLU(True))
#         self.class_classifier.add_module('c_drop1', nn.Dropout())
#         self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
#         self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
#         self.class_classifier.add_module('c_relu2', nn.ReLU(True))
#         self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
#         self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))
#
#         self.domain_classifier = nn.Sequential()
#         self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
#         self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
#         self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
#         self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
#         self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
#
#     def forward(self, input_data, alpha):
#         input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
#         feature = self.feature(input_data)
#         # view类似于reshape，一个参数定为-1，代表动态调整这个维度上的元素个数，以保证元素的总数保持不变
#         feature = feature.view(-1, 50 * 4 * 4)
#         reverse_feature = RevGrad.apply(feature, alpha)
#         class_output = self.class_classifier(feature)
#         domain_output = self.domain_classifier(reverse_feature)
#
#         return class_output, domain_output
