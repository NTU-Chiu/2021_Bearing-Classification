import torch
import torch.nn as nn

def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
        # nn.Sigmoid()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    # 2, 64, 512, 512
    # layers = layers + [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    # 2, 64, 256, 256

    return nn.Sequential(*layers) #  list-like layer

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
        # nn.Sigmoid()
    )
    return layer

class VGG16(nn.Module):
    def __init__(self, n_classes=2):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.max1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.max2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.max3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.max4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.max5 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # FC layers
        self.layer6 = vgg_fc_layer(16*16*512, 4096) # 長x寬xfeature 數
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Interpolation
        self.inter = nn.functional.interpolate(size=(512, 512), mode='bilinear') ###


        # Final layer
        self.layer8 = nn.Linear(4096, n_classes) # 後面我自己加的
        self.outputs = nn.Conv2d(384, 1, kernel_size=1, padding=0)



    # original VGG16
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        print(vgg16_features.shape)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out) # ([2, 4096])
        out = self.layer7(out)
        out = self.layer8(out)
        # (2, 2) --> ([2, 1, 512, 512])

        return  out # ,vgg16_features
    
if __name__== '__main__':
   x = torch.randn((2,3,512,512)) # batch_size, channel, image size, sizes
   f = VGG16()
   f(x)