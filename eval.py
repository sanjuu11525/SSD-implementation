from eval_utility import *
from torch.autograd import Variable
from data_coder import DataCoder
import torch.nn.functional as F

class Evaluation():
    def __init__(self, parameter, model):
        self.model = model
        self.datacoder = DataCoder()
        self.device = parameter['device']
    def evaluate(self, eval_loader):
                 
        gt_boxes = []
        gt_labels = []
        pred_boxes = []
        pred_scores = []
        pred_labels = []
        for it, (x_batch, ground_truth) in enumerate(eval_loader):
            box_gt, label_gt = ground_truth
            box_gt, label_gt = torch.FloatTensor(box_gt).squeeze(0), torch.LongTensor(label_gt).squeeze(0)
            gt_boxes.append(box_gt)
            gt_labels.append(label_gt)
            x_batch = Variable(x_batch.to(self.device))        
            pred_loc, pred_clf = self.model(x_batch)
        
            pred_loc, pred_clf = pred_loc.cpu(), pred_clf.cpu()
            pred_loc = pred_loc.squeeze(0)
            pred_clf = pred_clf.squeeze(0)
            pred_clf = F.softmax(pred_clf,dim=1)
            prediction = (pred_loc.data, pred_clf.data)
            decode_boxes, decode_scores, decode_labels = self.datacoder.targetDecoder(prediction)
            pred_boxes.append(decode_boxes.cpu())
            pred_scores.append(decode_scores.cpu())
            pred_labels.append(decode_labels.cpu())

        print voc_eval(pred_boxes, pred_labels, pred_scores,
                       gt_boxes, gt_labels, None,
                       iou_thresh=0.5, use_07_metric=True)
