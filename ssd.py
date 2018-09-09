import torch
import torch.nn as nn
from config import DATASET_CONFIG

class L2Norm(nn.Module):
    """L2Norm layer across all channels. Please respect the link for the origination.
    Reference:
      https://github.com/amdegroot/ssd.pytorch/blob/master/layers/modules/l2norm.py
    """
    def __init__(self, in_features, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters(scale)

    def reset_parameters(self, scale):
        nn.init.constant_(self.weight, scale)

    def forward(self, x):
        x = nn.functional.normalize(x, dim=1)
        scale = self.weight[None,:,None,None]
        return scale * x


class SSD(nn.Module):
    """Implementation of Single Shot MultiBox Object Detector.
    Arguments:
      pre_train_vgg16: Pre-training VGG16 model. Please see the reference on the github page.
    """
    def __init__(self, pre_train_vgg16=None):
        super(SSD, self).__init__()
        self.num_class = DATASET_CONFIG['num_classes']
        self.norm4 = L2Norm(512, 20)
        
        # build VGG16 
        vgg16 = self.buildVGG16()
        # initialize vgg16 
        if pre_train_vgg16 is None:
            vgg16 = self.initializeModule(vgg16)
            print('Initialization without pretraining VGG16')
        else:
            vgg16.load_state_dict(pre_train_vgg16)
            print('Initialization with pretraining VGG16')
        
        # set the extra layer output for vgg16
        vgg16_head, vgg16_tail = self.selectVggOut(vgg16, 22)
        self.vgg16_head = nn.ModuleList(vgg16_head)
        self.vgg16_tail = nn.ModuleList(vgg16_tail)
    
    
        # specify conv8, from 19x19 to 10x10
        Conv8 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(2, 2), padding=(1, 1)),
                          nn.ReLU(inplace=True))
        self.Conv8 = self.initializeModule(Conv8)
    
        # specify conv9, from 10x10 to 5x5
        Conv9 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(2, 2), padding=(1, 1)),
                          nn.ReLU(inplace=True))
        self.Conv9 = self.initializeModule(Conv9)

        # specify conv 10, from 5x5 to 3x3
        Conv10 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
                           nn.ReLU(inplace=True),
                           nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(0, 0)),
                           nn.ReLU(inplace=True))
        self.Conv10 = self.initializeModule(Conv10)
    
        # specify conv1x1
        Conv11 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
                           nn.ReLU(inplace=True),
                           nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(0, 0)),
                           nn.ReLU(inplace=True))
        self.Conv11 = self.initializeModule(Conv11)
    
        # the unmber of bounding boxes of each extra layer
        num_default_box = [4, 6, 6, 6, 4, 4]
    
        clf = []
        loc = []
        # prediction of location
        loc.append(nn.Conv2d(512 , 4 * num_default_box[0], kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)))
        loc.append(nn.Conv2d(1024, 4 * num_default_box[1], kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)))
        loc.append(nn.Conv2d(512 , 4 * num_default_box[2], kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)))
        loc.append(nn.Conv2d(256 , 4 * num_default_box[3], kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)))
        loc.append(nn.Conv2d(256 , 4 * num_default_box[4], kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)))
        loc.append(nn.Conv2d(256 , 4 * num_default_box[5], kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)))
        loc = nn.ModuleList(loc)
        self.loc = nn.Sequential(*loc)
        
        # prediction of classification
        clf.append(nn.Conv2d(512 , num_default_box[0] * self.num_class, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)))
        clf.append(nn.Conv2d(1024, num_default_box[1] * self.num_class, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)))
        clf.append(nn.Conv2d(512 , num_default_box[2] * self.num_class, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)))
        clf.append(nn.Conv2d(256 , num_default_box[3] * self.num_class, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)))
        clf.append(nn.Conv2d(256 , num_default_box[4] * self.num_class, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)))
        clf.append(nn.Conv2d(256 , num_default_box[5] * self.num_class, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)))
        clf = nn.ModuleList(clf)
        self.clf = nn.Sequential(*clf)
    
    def buildVGG16(self):
                          # Conv_1: 300x300
        vgg16 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                          nn.ReLU(inplace=True),
                          nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                          # Conv_2: 150x150
                          nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                          nn.ReLU(inplace=True),
                          nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                          # Conv_3: 75x75
                          nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                          nn.ReLU(inplace=True),
                          nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=True),
                          # Conv_4: 38x38
                          nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                          nn.ReLU(inplace=True),
                          nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                          # Conv_5: 19x19
                          nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                          nn.ReLU(inplace=True),
                          nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
                          # Conv_6 and 7: 19x19
                          nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(1024, 1024, kernel_size=1),
                          nn.ReLU(inplace=True)
                          )
        return vgg16
  
    def initializeModule(self, model):
      
        for m in model.children():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            
        return model 
    
    def forward(self, x):
        """Forward prpagation.
        Arguments:
          x: (tensor) batch image, sized [#batch, 3, 300, 300].
        Return:
          A tuple containing:
            1) loc_scope: (tensor) prediction of anchors, sized [#batch, 8732, 4].
            2) clf_scope: (tensor) prediction of class score for anchors, sized [#batch, 8732, num_class].
        """
        extra_input = []
        
        for elem in self.vgg16_head:
            x = elem(x)    
        extra_input.append(self.norm4(x))
        
        for elem in self.vgg16_tail:
            x = elem(x)  

        extra_input.append(x)
    
        x = self.Conv8(x)
        extra_input.append(x)
  
        x = self.Conv9(x)
        extra_input.append(x)
    
        x = self.Conv10(x)
        extra_input.append(x)
    
        x = self.Conv11(x)
        extra_input.append(x)

        loc_scope = []
        clf_scope = []
    
        batch, _, _, _ = x.size()  
        #process each branch
        for (loc_layer, clf_layer, ele_in) in zip(self.loc, self.clf, extra_input):
            loc_scope.append(loc_layer(ele_in).permute(0, 2, 3, 1).contiguous().view(batch, -1, 4))
            clf_scope.append(clf_layer(ele_in).permute(0, 2, 3, 1).contiguous().view(batch, -1, self.num_class))
      
        loc_scope = torch.cat(loc_scope, dim = 1)
        clf_scope = torch.cat(clf_scope, dim = 1)
    
        return (loc_scope, clf_scope)  
    
    def selectVggOut(self, vgg, loc):
        total_len = len(vgg)
        list_head = []
        for i in range(loc + 1):
            list_head.append(vgg[i])
      
        list_tail = []  
        for i in range(loc + 1, total_len):
            list_tail.append(vgg[i])
      
        return list_head, list_tail