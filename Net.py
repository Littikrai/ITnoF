
# PyTorch libraries and modules
from torch.nn import Linear, ReLU, Sequential, Conv2d, Conv1d, MaxPool2d, Module, BatchNorm2d, AvgPool2d

class Net(Module):
    def __init__(self):  # ก ำหนดโครงสร้ำงใน Constructor
        super(Net, self).__init__()
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 8, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            BatchNorm2d(8),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(8, 16, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            BatchNorm2d(16),
            ReLU(inplace=True),
            AvgPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers1 = Sequential(
            Linear(16 *13 * 27, 100)
        )

    # Defining how the data flows through these layers when performing the forward pass through the network
    def forward(self, x):
        x = self.cnn_layers(x)
        # convert feature map to vector form
        x = x.view(x.size(0), -1)
        x = self.linear_layers1(x)
        return x


