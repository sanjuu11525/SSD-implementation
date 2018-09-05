import torch
import torch.nn as nn
from torch.autograd import Variable
from config import VOC_CLASSES_LABEL_TO_ID
from config import DATASET_CONFIG

class MultiboxLoss(nn.Module):
    def __init__(self):
        super(MultiboxLoss, self).__init__()
        # the function returns a loss per batch element
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduce = False)
        # the function returns summed loss of a batch
        self.L1_loss = torch.nn.L1Loss(size_average = False)
    
    def forward(self, pred_loc, pred_clf, target_loc, target_cls): 
        '''Compute multibox loss. For regression loss, only possitive matches are involved.
           For classification loss, hard negative mining is used.
        Args:
          1)pred_loc: (tensor) the prediciton of bounding boxes, sized (#batch, 8723, 4).
          2)pred_clf: (tensor) the prediciton of category, sized (#batch, 8723, num_class).
          3)target_loc: (tensor) the target of bounding boxes, sized (#batch, 8723, 4).
          4)target_cls: (tensor) the target of category, sized (#batch, 8723, 1).
        Return:
          1)loss_for_cls: (tensor) CrossEntropyLoss for classification.
          2)loss_for_loc: (tensor) L1Loss for regression.
        '''
        # a mask for positive matches, sized (batch, 8723, 1)
        pos_matches = (target_cls != VOC_CLASSES_LABEL_TO_ID['background'])
        # a mask for negative matches, sized (batch, 8723, 1)
        neg_matches = (target_cls == VOC_CLASSES_LABEL_TO_ID['background'])
        
        num_pos_match = pos_matches.sum()
        num_neg_match = neg_matches.sum()

        # if the number of positive matches is zero, return
        if(num_pos_match == 0):
            return Variable(torch.FloatTensor([0.0]).cuda(),requires_grad=True), Variable(torch.FloatTensor([0.0]).cuda(),requires_grad=True)
        
        # adapte the mask for location, sized (batch, 8732, 4)
        matches_for_loc = pos_matches.clone().expand_as(target_loc)
        # execute the mask for target and reshape with size by (batch, 8732, 4)->(num_pos_match, 4)
        target_loc = target_loc[matches_for_loc].view(-1, 4)
        target_loc = Variable(target_loc)
        # execute the mask for pred and reshape with size by (batch, 8732, 4)->(num_pos_match, 4)
        pred_loc = pred_loc[matches_for_loc].view(-1, 4)
        # compute L1 loss for location
        loss_for_loc  = self.L1_loss(pred_loc, target_loc)
        # make average
        loss_for_loc /= (float(num_pos_match))

        num_class = DATASET_CONFIG['num_classes']
        
        # positive training examples
        target_pos_cls  = target_cls[pos_matches]
        target_pos_cls  = Variable(target_pos_cls)
        # adapte the mask with size batch, 8723, num_class)
        matches_for_cls = pos_matches.clone().expand_as(pred_clf)
        # execute the mask for pred and reshape with size by (batch, 8732, num_class)->(num_pos_match, num_class)
        pred_pos_clf    = pred_clf[matches_for_cls].view(-1, num_class)
        # compute cross entropy for classification
        loss_for_pos_cls = self.cross_entropy_loss(pred_pos_clf, target_pos_cls)
       
        # negative training example
        target_neg_cls = target_cls[neg_matches]
        target_neg_cls = Variable(target_neg_cls)
        matches_for_pos_cls = neg_matches.clone().expand_as(pred_clf)
        pred_neg_clf     = pred_clf[matches_for_pos_cls].view(-1, num_class)
        loss_for_neg_cls = self.cross_entropy_loss(pred_neg_clf, target_neg_cls)
        
        # hard negative mining
        _, indices    = loss_for_neg_cls.sort(descending=True)
        # compute how many negative matches involved
        num_neg_match = num_pos_match * 3 if (num_pos_match * 3) < num_neg_match else num_neg_match
        # extract the corresponding for training
        loss_for_neg_cls = loss_for_neg_cls[indices[:num_neg_match]]
        # compute the final classification loss   
        loss_for_cls = (loss_for_pos_cls.sum() + loss_for_neg_cls.sum()) / float(num_pos_match + num_neg_match)
 
        return loss_for_cls, loss_for_loc