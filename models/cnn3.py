import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class Net3(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions

    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        Args:
            params: (Params) contains num_channels
        """
        super(Net3, self).__init__()
        self.num_channels = 32

        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels * 4, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels * 4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(4 * 4 * self.num_channels * 4, 10)
        self.dropout_rate = 0.5

    def forward(self, x):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 32 x 32 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        ### x.shape = torch.Size([128, 3, 32, 32])
        #                                                  -> batch_size x 3 x 32 x 32
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3

        x = self.bn1(self.conv1(x))  # batch_size x num_channels x 32 x 32
        ### torch.Size([128, 32, 32, 32])
        x = F.relu(F.max_pool2d(x, 2))  # batch_size x num_channels x 16 x 16
        ### torch.Size([128, 32, 8, 8])
        x = self.bn2(self.conv2(x))  # batch_size x num_channels*2 x 16 x 16
        x = F.relu(F.max_pool2d(x, 2))  # batch_size x num_channels*2 x 8 x 8
        ### torch.Size([128, 128, 8, 8])
        # flatten the output for each image
        x = x.view(-1, 4 * 4 * self.num_channels * 4)  # batch_size x 4*4*num_channels*4

        # apply 1 fully connected layers with dropout
        # s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
        #               p=self.dropout_rate, training=self.training)  # batch_size x self.num_channels*4
        x = self.fc1(x)
        #torch.Size([512, 10])
        return x

# def t():
#     net = Net1()
#     y = net(torch.randn(2,3,32,32))
#     print(y.size())