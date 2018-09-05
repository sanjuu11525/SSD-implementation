import torch
import numpy as np
from itertools import product as product
from config import SSD300_DEFAULTBOX_CONFIG
from config import DATA_CODER_CONFIG
from config import VOC_CLASSES_LABEL_TO_ID
from config import DATASET_CONFIG

class DataCoder():
    def __init__(self):
        self.default_box = self.defaultBox()
        self.variance = [0.1, 0.2]
        
    def defaultBox(self):
        """Generate default bounding boxes. The default boxes is formulated by [cx, cy, w, h] firstly.
        Return:
          default_box: (tensor) default anchors with normalized [xmin, ymin, xmax, ymax], sized [8732, 4].
        """
        steps      = SSD300_DEFAULTBOX_CONFIG['steps']
        image_size = SSD300_DEFAULTBOX_CONFIG['image_size']
        min_sizes  = SSD300_DEFAULTBOX_CONFIG['min_sizes']
        max_sizes  = SSD300_DEFAULTBOX_CONFIG['max_sizes']
        aspect_ratios    = SSD300_DEFAULTBOX_CONFIG['aspect_ratios']
        extra_layer_size = SSD300_DEFAULTBOX_CONFIG['extra_layer_size']
                
        bbox = []
        for k, f in enumerate(extra_layer_size):
            for i, j in product(range(f), repeat=2):
                # compute the factor for normalization
                f_k = image_size / steps[k]               
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                # rel size: min_size. aspect_ratio: 1
                s_k = min_sizes[k] / image_size
                bbox += [cx, cy, s_k, s_k]               
                # rel size: sqrt(s_k * s_(k+1)). aspect_ratio: 1
                s_k_prime = np.sqrt(s_k * (max_sizes[k] / image_size))
                bbox += [cx, cy, s_k_prime, s_k_prime]
                # rest of aspect ratios
                for ar in aspect_ratios[k]:
                    bbox += [cx, cy, s_k * np.sqrt(ar), s_k / np.sqrt(ar)]
                    bbox += [cx, cy, s_k / np.sqrt(ar), s_k * np.sqrt(ar)]
        
        # reshape to [8723, 4] as default setting
        default_box = torch.Tensor(bbox).view(-1, 4)
        
        # convert to [xmin, ymin, xmax, ymax], consistent with ground truth
        default_box = torch.cat((default_box[:, :2] - default_box[:, 2:]/2, 
                                 default_box[:, :2] + default_box[:, 2:]/2), 1)
        # clamp boxes for boundary cells
        default_box.clamp_(max=1.0, min=0.0)
        
        return default_box    
  
    def IoU(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          iou: (tensor) sized [N,M].
        Reference:
          https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
        '''
        N = box1.size(0)
        M = box2.size(0)
        
        box1_ = box1.clone()
        box2_ = box2.clone()

        lt = torch.max(
            box1_[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2_[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1_[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2_[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1_[:,2]-box1_[:,0]) * (box1_[:,3]-box1_[:,1])  # [N,]
        area2 = (box2_[:,2]-box2_[:,0]) * (box2_[:,3]-box2_[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou
      
    def nms(self, conf, loc):
        '''This function executes non maximum suppression. The nms is based on boxes with same labels.
        Args:
          1)conf: (tensor) ,sized [#box] before nms.
          2)loc: (tensor) ,sized [#box, 4] before nms.
        Return:
          1)conf: (tensor) ,sized [#box] after nms.
          2)loc: (tensor) ,sized [#box, 4] after nms.
        '''
        nms_threshold = DATA_CODER_CONFIG['nms_threshold']
        _, indices = conf.sort(descending=True)

        keep = []
        while(True):
            max_val_idx = indices[0]
            keep.append(max_val_idx)
            # extract current box
            cur_box   = torch.index_select(loc, 0, max_val_idx)
            # extract other boxes 
            other_box = torch.index_select(loc, 0, indices)
            # compute the iou
            iou = self.IoU(cur_box, other_box)
            # only box(es) not be suppressed are preserved
            preserve = iou.le(nms_threshold)
            if not preserve.any():
                break 
            indices = torch.masked_select(indices, preserve)
            
        keep = torch.LongTensor(keep)
        conf = torch.index_select(conf, 0, keep)
        loc  = torch.index_select(loc, 0, keep)
        
        return conf, loc
       
    def targetDecoder(self, prediction, conf_threshold = DATA_CODER_CONFIG['conf_threshold']):
        '''Decode the prediction to the ground truth format. Batch is restrictared to one.
        Args:
          A tuple containing:
            pred_loc: (tensor) the prediction of bounding boxes, sized [8732,4].
            pred_clf: (tensor) the prediction of class, sized [8732, 21].
        Return:
          1)output_box: (tensor) the number of bbox after the suppression, sized [#box].
          2)output_conf: (tensor) confidence to each box, sized [#box].
          3)output_label: (tensor) label to each box, sized [#box].
        '''
        # unpack 
        pred_loc, pred_clf = prediction
        
        # transform to [xmin, ymin, xmax, ymax]        
        cx_cy = pred_loc[:, :2] * (self.variance[0] * (self.default_box[: ,2:] - self.default_box[:, :2]))
        cx_cy = cx_cy + (self.default_box[:,:2] + self.default_box[:,2:])/2.0
        w_h   = torch.exp(pred_loc[:, 2:] * self.variance[1]) * (self.default_box[:, 2:] - self.default_box[:, :2])
        eval_loc = torch.cat([cx_cy - w_h/2.0, cx_cy + w_h/2.0], dim = 1)
        
        # skip the background
        num_class_ = DATASET_CONFIG['num_classes'] - 1
        
        output_box   = []
        output_conf  = []
        output_label = []
        
        # use conf_threshold for each class to decide corresponding bbox 
        for ith_class in range(num_class_):
            # extract conf of all boxes
            ith_conf  = pred_clf[:, ith_class]
            # create a mask of confidence
            mask_conf = ith_conf.ge(conf_threshold)
            if not mask_conf.any():
                continue
            # adapte mask_conf to the dim of location    
            mask_loc = mask_conf.unsqueeze(1).expand_as(eval_loc)
            # masking for both
            ith_conf = ith_conf[mask_conf]
            ith_loc  = eval_loc[mask_loc].view(-1, 4)
            # execute non maximum suppression
            conf, box = self.nms(ith_conf, ith_loc)
            output_box.append(box)
            output_conf.append(conf)
            output_label.append(torch.LongTensor(len(box)).fill_(ith_class))
        
        if 0 == len(output_box):
            print("Decoder decodes nothing")
            output_box.append(torch.FloatTensor([[0, 0, 0, 0]]))
            output_label.append(torch.LongTensor([20]))
            output_conf.append(torch.FloatTensor([0.]))

        output_box   = torch.cat(output_box,   0)
        output_label = torch.cat(output_label, 0)
        output_conf  = torch.cat(output_conf,  0)
        
        return output_box, output_conf, output_label     
        
    def targetCoder(self, target):
        '''Encode the ground truth to the target format. Return encoded for loss computation.
        Match the default bounding box/es to the ground truth and generate the target for the loss function.
        Args:
          A tuple containing:
            1)batch_box_gt: (list) bounding boxes gt, len():#batch, dim(ele):2.
            2)batch_label_gt: (list) labels gt, len():#batch, dim(ele):1.
        Return:
          1)batch_target_loc: (tensor) target location, sized (#batch, 8723, 4).
          2)batch_target_cls: (tensor) target label, sized (#batch, 8723, 1).
        '''
        batch_box_gt, batch_label_gt = target
        iou_threshold = DATA_CODER_CONFIG['iou_threshold']
        batch_target_loc = []
        batch_target_cls = []
        
        # unpack ground truth for each image
        for (box_gt, label_gt) in zip(batch_box_gt, batch_label_gt):
            
            # box_gt: [xmin, ymin, xmax, ymax], label_gt: [label]
            box_gt, label_gt = torch.FloatTensor(box_gt), torch.LongTensor(label_gt).unsqueeze_(1)
            # compute IoU between gt boxes and default anchors, sized (#obj, 8732)
            iou = self.IoU(box_gt, self.default_box)   
            iou_max, idx_max = iou.max(0)
            box_gt = box_gt[idx_max]
            
            # box coordinate transformation. according to the paper section: Training objective
            cx_cy = (box_gt[:,:2] + box_gt[:,2:]) / 2.0 - (self.default_box[:,:2] + self.default_box[:,2:]) / 2.0            
            cx_cy = cx_cy / (self.variance[0] * (self.default_box[:,2:] - self.default_box[:,:2]))          
            w_h = (box_gt[:,2:] - box_gt[:,:2])/(self.default_box[:,2:] - self.default_box[:,:2])
            w_h = torch.log(w_h) / self.variance[1]             
            target_loc = torch.cat([cx_cy, w_h], 1)
            
            batch_target_loc.append(target_loc)
            conf_target = label_gt[idx_max]
            
            # set iou of boxes below the threshold to background
            conf_target[iou_max.le(iou_threshold)] = VOC_CLASSES_LABEL_TO_ID['background']
            batch_target_cls.append(conf_target)

        # output targets of the location and classification    
        batch_target_loc = torch.stack(batch_target_loc, 0) 
        batch_target_cls = torch.stack(batch_target_cls, 0)

        return batch_target_loc, batch_target_cls