# domain classifier，生成feature map，使用交叉熵分类
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
from torch.autograd import Variable,Function

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


class VGG(nn.Module):
	def __init__(self, features):
		super(VGG, self).__init__()
		self.features = features
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 1000),
		)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x


class extractor(nn.Module):
	def __init__(self, pretrained):
		super(extractor, self).__init__()
		vgg16_bn = VGG(make_layers(cfg, batch_norm=True))
		if pretrained:
			vgg16_bn.load_state_dict(torch.load('./pths/vgg16_bn-6c64b313.pth'))
		self.features = vgg16_bn.features

	def forward(self, x):
		out = []
		for m in self.features:
			x = m(x)
			if isinstance(m, nn.MaxPool2d):
				out.append(x)

		return out[1:]


class merge(nn.Module):
    def __init__(self):
        super(merge, self).__init__()

        self.conv1 = nn.Conv2d(1024, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(384, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(192, 32, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def one_forward(self, x):
        y = F.interpolate(x[3], scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[2]), 1)
        y = self.relu1(self.bn1(self.conv1(y)))
        y = self.relu2(self.bn2(self.conv2(y)))

        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[1]), 1)
        y = self.relu3(self.bn3(self.conv3(y)))
        y = self.relu4(self.bn4(self.conv4(y)))

        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[0]), 1)
        y = self.relu5(self.bn5(self.conv5(y)))
        y = self.relu6(self.bn6(self.conv6(y)))

        y = self.relu7(self.bn7(self.conv7(y)))
        return y

    def forward(self, x):
        y = self.one_forward(x)
        return y

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        #pdb.set_trace()
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)

class netD(nn.Module):
    def __init__(self, context=False):
        super(netD, self).__init__()
        # self.conv1 = conv3x3(1024, 512, stride=2)
        # self.bn1 = nn.BatchNorm2d(512)
        # self.conv2 = conv3x3(512, 128, stride=2)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.conv3 = conv3x3(128, 128, stride=2)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.fc = nn.Linear(128,2)
        self.conv1 = conv3x3(32, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = conv3x3(64, 128)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 256)
        self.bn3 = nn.BatchNorm2d(256)
        # self.fc = nn.Linear(128, 2)
        self.conv4 = conv3x3(256,1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)

        x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
        x = self.conv4(x)

        return x


class output(nn.Module):
    def __init__(self, scope=512):
        super(output, self).__init__()
        self.conv1 = nn.Conv2d(32, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(32, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(32, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.scope = 512
        self.netD = netD()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, target=False):
        if target == False:
            score = self.sigmoid1(self.conv1(x))
            loc   = self.sigmoid2(self.conv2(x)) * self.scope
            angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi
            geo   = torch.cat((loc, angle), 1)
            domain_cls = self.netD(grad_reverse(x))
            return score, geo, domain_cls

        domain_cls = self.netD(grad_reverse(x))
        return domain_cls


class EAST(nn.Module):
    def __init__(self, pretrained=True):
        super(EAST, self).__init__()
        self.extractor = extractor(pretrained)
        self.merge     = merge()
        self.output    = output()

    def forward(self, x, target=False):
        x = self.extractor(x)
        x= self.merge(x )
        return self.output(x, target)


if __name__ == '__main__':
    m = EAST(False)
    # print(m)
    x = torch.randn(1, 3, 256, 256)
    score, geo , cls= m(x)

    print('score size:', score.size())
    print('geo size:', geo.size())
    print('cls size:', cls.size())


