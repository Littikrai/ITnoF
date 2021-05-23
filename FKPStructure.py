from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, AvgPool2d

class FKPStructure(Module):
    def __init__(self):
        super(FKPStructure, self).__init__()
        #Have 3 Covolution layer for feature extraction
        self.cnn_layers = Sequential(
            #first Convolution layer and Max pooling
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            #second Convolution layer and Max pooling
            Conv2d(4, 8, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            BatchNorm2d(8),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            #third Convolution layer and Average pooling
            Conv2d(8, 16, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            BatchNorm2d(16),
            ReLU(inplace=True),
            AvgPool2d(kernel_size=2, stride=2),
            #output is (16 *13.5 * 27.5)
        )
        #Have 1 Fully Connected layer for classification
        self.linear_layers1 = Sequential(
            Linear(16 *13 * 27, 100)
        )

    #data flow
    def forward(self, x):
        x = self.cnn_layers(x)
        # convert feature map to vector form
        x = x.view(x.size(0), -1)
        x = self.linear_layers1(x)
        return x
