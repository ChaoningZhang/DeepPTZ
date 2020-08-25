import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import pdb
#from SE_module import SELayer
try:
    from spatial_correlation_sampler import spatial_correlation_sample
except ImportError as e:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn("failed to load custom correlation module"
                      "which is needed for FlowNetC", ImportWarning)

__all__ = ['MyInception3', 'myinception_v3', 'MyInception3_siamese']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

class MyInception3_siamese(nn.Module):
    def __init__(self, siamese=True, pretrained = True, pretrained_fixed=True,aux_logits=False):
        super(MyInception3_siamese, self).__init__()

        self.model = MyInception3_siamese_single(siamese=siamese, pretrained=pretrained, pretrained_fixed=pretrained_fixed, aux_logits=aux_logits)

    def forward(self, input1, input2):
        #pdb.set_trace()
        x1 = self.model(input1, input2)
        x2 = self.model(input2, input1)
        x         = torch.cat((x1, x2), 1)
        return x

class MyInception3_siamese_single(nn.Module):
    def __init__(self, siamese=True, pretrained = True, pretrained_fixed=True,aux_logits=False):
        super(MyInception3_siamese_single, self).__init__()
        input_channels = 3
        num_classes = 3

        num_classes_extra = 4
        if not siamese:
            input_channels = 6
            pretrained = False 
        self.feature = myinception_v3(pretrained=pretrained, pretrained_fixed=pretrained_fixed, input_channels=input_channels, aux_logits=aux_logits)
        self.relation= CorreRelation(288, num_classes+num_classes_extra)
        self.FeatureCorrelation= FeatureCorrelation()
        self.siamese = siamese
        #self.conv_redir = BasicConv2d(288, 32, kernel_size=1, padding=0)
        self.conv_all = BasicConv2d(35*35+int(0/2), 288, kernel_size=1, padding=0)
        #self.SELayer = SELayer(225, 15)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_once(self, x):
        return  self.feature(x)
    def forward(self, input1, input2):
        #pdb.set_trace()
        if self.siamese:
            feature1 = self.forward_once(input1)
            feature2 = self.forward_once(input2)

            out_correlation1 = self.FeatureCorrelation(feature1, feature2)
 

            feature_f = self.conv_all(torch.cat([out_correlation1], dim=1))
            x = self.relation(feature_f)
        else:
            feature = self.forward_once(torch.cat((input1, input2), 1))
            #feature  = torch.cat((feature1, feature2), 1) #8 x 8 x 2048*2
             # .detach()
        return x

class Relation(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(Relation, self).__init__()
        self.fc = nn.Linear(in_channels, num_classes)
        self.Mixed_6a = InceptionB(in_channels)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        # if aux_logits:
        #     self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        #self.fc.stddev = 0.001

    def forward(self, x):
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 3
        return x

class CorreRelation(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(CorreRelation, self).__init__()
        # self.conv0 = BasicConv2d(in_channels, 64, kernel_size=3, padding=1)
        factor=1
        self.fc = nn.Linear(int(2048/factor), num_classes)
        #self.fc.stddev = 0.001
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        # if aux_logits:
        #     self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
    def forward(self, x):
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 4096
        # x = self.conv0(x)
        # # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 3
        return x

class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
#        print(feature.size())
#        print(torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).size())
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)
    
class FeatureCorrelation(torch.nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()
    
    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
        feature_B = feature_B.view(b,c,h*w).transpose(1,2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B,feature_A)
        correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        return correlation_tensor

def correlate(input1, input2):
    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=15,
                                          stride=1,
                                          padding=0,
                                          dilation_patch=1)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
    return F.relu_(out_corr)

def correlateleaky(input1, input2):
    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=15,
                                          stride=1,
                                          padding=0,
                                          dilation_patch=1)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
    return F.leaky_relu_(out_corr, 0.1)


def myinception_v3(pretrained=False, pretrained_fixed=True, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        model = MyInception3(**kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']), strict=False)
        if pretrained_fixed:
            for param in model.parameters():
                param.requires_grad = False
            return model

    return MyInception3(**kwargs)


class MyInception3(nn.Module):

    def __init__(self, input_channels = 3, aux_logits=False, transform_input=False):
        super(MyInception3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2df(input_channels, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2df(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2df(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2df(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2df(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        # self.Mixed_6a = InceptionB(288)
        # self.Mixed_6b = InceptionC(768, channels_7x7=128)
        # self.Mixed_6c = InceptionC(768, channels_7x7=160)
        # self.Mixed_6d = InceptionC(768, channels_7x7=160)
        # self.Mixed_6e = InceptionC(768, channels_7x7=192)
        # if aux_logits:
        #     self.AuxLogits = InceptionAux(768, num_classes)
        # self.Mixed_7a = InceptionD(768)
        # self.Mixed_7b = InceptionE(1280)
        # self.Mixed_7c = InceptionE(2048)
        #self.fc = nn.Linear(2048, num_classes)
        #self.fc1 = nn.Linear(2048, 3)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         import scipy.stats as stats
        #         stddev = m.stddev if hasattr(m, 'stddev') else 0.1
        #         X = stats.truncnorm(-2, 2, scale=stddev)
        #         values = torch.Tensor(X.rvs(m.weight.data.numel()))
        #         values = values.view(m.weight.data.size())
        #         m.weight.data.copy_(values)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        # x = self.Mixed_6a(x)
        # # 17 x 17 x 768
        # x = self.Mixed_6b(x)
        # # 17 x 17 x 768
        # x = self.Mixed_6c(x)
        # # 17 x 17 x 768
        # x = self.Mixed_6d(x)
        # # 17 x 17 x 768
        # x = self.Mixed_6e(x)
        # 17 x 17 x 768
        # if self.training and self.aux_logits:
        #     aux = self.AuxLogits(x)
        # 17 x 17 x 768
        # x = self.Mixed_7a(x)
        # # 8 x 8 x 1280
        # x = self.Mixed_7b(x)
        # # 8 x 8 x 2048
        # x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        #x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        #x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        #x = x.view(x.size(0), -1)
        # 2048
        #x = self.fc(x)
        #x = self.fc1(x)
        # 1000 (num_classes)
        # if self.training and self.aux_logits:
        #     return x, aux
        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2df(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2df(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2df(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2df(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2df(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2df(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2df(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x
class BasicConv2df(nn.Module):
    # f: feature extraction
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2df, self).__init__()
        factor=1
        if in_channels%2 == 0:
            in_channels = int(in_channels/factor)
        out_channels = int(out_channels/factor)
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        #self.conv3 = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        #self.gn = nn.GroupNorm(int(16/factor), out_channels)
        self.relu = nn.ReLU(inplace=True)
        #self.bn1 = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x): 
        x = self.conv(x) #+ self.conv3(0.1*x*x.abs())
        x = self.bn(x)
        x = self.relu(x)
        #x = self.gn(x)  
        return x # F.relu(x, inplace=True)

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        factor=1
        if in_channels%2 == 0:
            in_channels = int(in_channels/factor)
        out_channels = int(out_channels/factor)
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        #self.conv3 = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        #self.gn = nn.GroupNorm(int(16/factor), out_channels)
        self.relu = nn.ReLU(inplace=True)
        #self.bn1 = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x): 
        x = self.conv(x) #+ self.conv3(0.1*x*x.abs())
        x = self.bn(x)
        x = self.relu(x)
        return x # F.relu(x, inplace=True)
