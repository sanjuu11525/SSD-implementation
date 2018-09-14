import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import VOC_CLASSES_LABEL_TO_ID
from config import VOC_CLASSES_ID_TO_LABEL

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
    
class VOCDataBase():
    """Data container for Pascal VOC dataset. All images, annotations are handled in this class. 
    The augmentation pipeline is associated by a parameter set.
    Arguments:
      parameter: Parameter set for dataset configuration and augmented data.
    """
    def __init__(self, parameter):
        root = parameter['root']
        image_set = parameter['image_set']
        year_set = parameter['years']
        if year_set[0] == year_set[1]:
            raise ValueError('Duplicate yeara in the parameter set')
        self.transform = parameter['transforms']
        self.keep_difficult = parameter['keep_difficult']

        self.ids = []
        self._annopath = []
        self._imgpath = []
        for year in year_set:
            dataset = 'VOC' + year
            imgsetpath = os.path.join(root, dataset, 'ImageSets', 'Main', '%s.txt')
            for line in open(imgsetpath % image_set):
                self.ids.append(line.strip())
                self._annopath.append(os.path.join(root, dataset, 'Annotations', '%s.xml'))
                self._imgpath.append(os.path.join(root, dataset, 'JPEGImages', '%s.jpg'))
  
    def getAnnotations(self, target):
        """Receive annotations from annotation files per image.
        Arguments:
          target: File object.
        Return:
          res: (list) containing all annotations with [[xmin, ymin, xmax, ymax, label_ind], ... ].
               len(res) is annotated images. res[i] provides annotated bounding box(es) and the category.
        """
        res = []
        # loop all 'object' per image
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            label_idx = VOC_CLASSES_LABEL_TO_ID[name]
            # [xmin, ymin, xmax, ymax, label_ind]
            bndbox.append(label_idx)
            res += [bndbox]  

        return res
    def pull_image(self, index):
        """Fetch image with BGR format by OpenCV i/o.
        Arguments:
          image_id: scalar.
        Return:
          img: (numpy.narray) image, sized [height, width, 3]. 
        """
        img = cv2.imread(self._imgpath[index] % self.ids[index], cv2.IMREAD_COLOR)
        
        return img
    def pull_annotation(self, index):
        """Fetch annotation by xml parser.
        Arguments:
          image_id: scalar.
        Return:
          1) gt_box: (list) ground truth of normalized bounding box coordinates, sized [#box, 4].
          2) gt_id: (list) ground truth of labeled bounding, sized [#box].  
        """
        ground_truth_root = ET.parse(self._annopath[index] % self.ids[index]).getroot()
        ground_truth = self.getAnnotations(ground_truth_root)
        
        gt_box = []
        gt_id = []
        num_obj = len(ground_truth)
        for the_ith_obj in range(num_obj):
            gt_box.append(ground_truth[the_ith_obj][:-1])
            gt_id.append(ground_truth[the_ith_obj][-1])
            
        return gt_box, gt_id
        
    def __getitem__(self, index):
        """Access the image and annotation based on the index. Return augumented data if required.
        Arguments:
          index: scalar.
        Return:
          1) img: (torch) image, sized [3, 300, 300].
          2) gt_box: (list) ground truth of normalized bounding box coordinates, sized [#box, 4].
          3) gt_id: (list) ground truth of labeled bounding, sized [#box].  
        """
        img = self.pull_image(index)
        gt_box, gt_id = self.pull_annotation(index)
    
        if self.transform is not None:
            img, gt_box, gt_id = self.transform(img, gt_box, gt_id)
        
        return img, gt_box, gt_id

    def __len__(self):
        return len(self.ids)

    def showGroundTruth(self, index):
        """Visualize the image with ground truth.
        Arguments:
          index: (scalar).
        """
        img_id = self.ids[index]
        img = self.pull_image(index)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_box, gt_id = self.pull_annotation(index)

        plt.figure(figsize=(10, 10))
        colors = plt.cm.gnuplot2(np.linspace(0, 1, 21)).tolist()
        plt.imshow(img)
        currentAxis = plt.gca()
        
        for box, id_ in zip(gt_box, gt_id):
            x_min = box[0]
            y_min = box[1]
            width = box[2] - x_min 
            height = box[3] - y_min 
            color = colors[id_]
            display_txt = 'Id: %d, %s'%(id_, VOC_CLASSES_ID_TO_LABEL[int(id_)])
            currentAxis.add_patch(plt.Rectangle([x_min, y_min], width, height, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(x_min, y_min, display_txt, bbox={'facecolor':color, 'alpha':0.5})
            
    def showPredictedData(self, index, pred_loc, pred_score, pred_id):
        """Visualize the image with predictions.
        Arguments:
          index: (scalar).
          pred_loc: (list), predicted bbox locations.
          pred_score: (list), predicted scores.
          pred_id: (list), predicted id of objets.
        """
        img = self.pull_image(index)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        plt.figure(figsize=(10, 10))
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, 21)).tolist()
        plt.imshow(img)
        currentAxis = plt.gca()
        
        for box, score, id_ in zip(pred_loc, pred_score, pred_id):
            x_min = box[0] * w
            y_min = box[1] * h
            width = box[2] * w - x_min
            height = box[3] * h - y_min
            color = colors[id_]
            label = VOC_CLASSES_ID_TO_LABEL[int(id_)]
            display_txt = 'Id: %d, %s, Score: %.2f'%(id_, label, score)
            currentAxis.add_patch(plt.Rectangle([x_min, y_min], width, height, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(x_min, y_min, display_txt, bbox={'facecolor':color, 'alpha':0.5})

