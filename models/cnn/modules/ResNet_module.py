from . import *

def conv_1d(in_planes, out_planes, kernel_size=3, stride=1, groups=1, padding=1, dilation=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)

def conv1x1_1d(in_planes, out_planes, stride=1): # we use this function when we have to downsampling
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride,
                    bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, block_kernel_size=3, padding=1,downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,dropout_p=0.,use_batchnorm=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # if dilation > 1:
        #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.dilation = dilation
        self.use_batchnorm = use_batchnorm
        print(f'block_kernel_size == {block_kernel_size}')
        self.conv1 = conv_1d(in_planes=inplanes, out_planes=planes, kernel_size=block_kernel_size, stride=stride,
                             groups=groups,padding=padding,dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_1d(in_planes=planes, out_planes=planes, kernel_size=block_kernel_size, groups=groups,
                             padding=padding,dilation=dilation)
        
        
        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()
        if self.use_batchnorm:
            self.bn1 = norm_layer(planes)
            self.bn2 = norm_layer(planes)

        self.dropout_p = dropout_p

        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()

        if self.dropout_p > 0:
            self.dropout1 = nn.Dropout(p=self.dropout_p)
            self.dropout2 = nn.Dropout(p=self.dropout_p)

        self.block_kernel_size = block_kernel_size
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # print('kernel size : ',self.block_kernel_size)
        # print('dilation : ', self.dilation)
        identity = x

        out = self.conv1(x)
        # print('out1.shape : ',out.shape )
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout1(out)

        out = self.conv2(out)
        # print('out2.shape : ',out.shape)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        # print('out : ',out.shape)
        # print('id : ',identity.shape)
        out += identity
        out = self.relu(out)

        out = self.dropout2(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        block_kernel_size: int =3,
        padding:int =1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,dropout_p:float=0.,use_batchnorm:bool=True):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        print(f'block_kernel_size == {block_kernel_size}')
        print(f'padding == {padding}')
        print(f'stride = {stride}')
        print(f'dilation = {dilation}')
        self.conv1 = conv1x1_1d(in_planes=inplanes, out_planes=width, stride=1)
        
        self.conv2 = conv_1d(in_planes=width, out_planes=width,kernel_size=block_kernel_size, stride=stride,padding=padding, groups=groups, dilation=dilation)
        
        self.conv3 = conv1x1_1d(in_planes=width, out_planes=planes * self.expansion, stride=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.bn1 = norm_layer(width)
            self.bn2 = norm_layer(width)
            self.bn3 = norm_layer(planes * self.expansion)

        self.dropout_p = dropout_p

        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()
        self.dropout3 = nn.Identity()
        if self.dropout_p > 0:
            self.dropout1 = nn.Dropout(p=self.dropout_p)
            self.dropout2 = nn.Dropout(p=self.dropout_p)
            self.dropout3 = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # print(out.shape)

        out = self.dropout2(out)


        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        # print(f'out shape = {out.shape} // identity shape = {identity.shape}')
        out += identity
        out = self.relu(out)

        out = self.dropout3(out)

        return out