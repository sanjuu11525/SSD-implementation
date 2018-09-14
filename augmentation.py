import torch
import numbers
import random
import cv2
import numpy as np

class Compose(object):
    """Composes several transforms together. This pipeline is customized to data preparation of object detection. 
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Reference:
        https://github.com/pytorch/vision/tree/master/torchvision/transforms
    Example:
    >>> Compose([[RandomCrop(300), RandomHorizontalFlip(), NormalizeBoundingBox(), Resize(300)])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes, labels):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ToTensor(object):
    """Convert a BGR image(numpy.ndarray) to tensor. Converts a numpy.ndarray (H x W x C) with integer type
    to a torch.FloatTensor of shape (C x H x W)
    """
    def __call__(self, img, boxes, labels):
        """
        Args:
            img (numpy.ndarray): Image to be converted to tensor.
            boxes: Bounding boxes of the image.
            labels: Labels of bounding boxes.
        Returns:
            Tensor: Converted image.
            boxes: No operation.
            labels: No operation.
        """
        img = img.astype(np.float32)
        return torch.from_numpy(img).permute(2, 0, 1), boxes, labels

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class RandomCrop(object):
    """Crop the given image(numpy.ndarray) at a random location. 
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), the sequence (size, size) is
            created.
        p (float): Probability of executing the random crop.
        th (float): Threshold to decide whether the ground truth is discarded after cropping.   
    """
    def __init__(self, size, p=0.5, th=0.5):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        
        self.p = p
        self.th = th
    def get_params(self, img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (numpy.ndarray): Image for info of size.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) patch of image after cropping randomly.
        """
        h, w, _ = img.shape
        th, tw = output_size
        if w <= tw or h <= th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw
    def __call__(self, img, boxes, labels):
        """
        Args:
            img (numpy.ndarray): Image to be cropped.
            boxes: Bounding boxes of the image.
            labels: Labels of bounding boxes.
        Returns:
            img: Cropped image.
            boxes: Preserved boxes after cropping img.
            labels: Preserved labels of boxes.
        """
        if random.random() < self.p:
            return img, boxes, labels
        
        i, j, h, w = self.get_params(img, self.size)
        crop_img = img[i:i+h, j:j+w, :]

        preserved_boxes = []
        preserved_labels = []
        for box, label in zip(boxes, labels):
            if (box[0] < (j+w)) and (box[2] > j):
                cropped_box = list(box)
                cropped_box[0] = cropped_box[0] - j if cropped_box[0] > j else 0
                cropped_box[2] = cropped_box[2] - j if (j + w) > cropped_box[2] else w - 1
                cropped_box[1] = cropped_box[1] - i if cropped_box[1] > i else 0
                cropped_box[3] = cropped_box[3] - i if (i + h) > cropped_box[3] else h - 1
                box_width = float(box[2] - box[0])
                cropped_width = float(cropped_box[2] - cropped_box[0])
                if cropped_width/box_width > self.th: 
                    preserved_boxes.append(cropped_box)
                    preserved_labels.append(label)
         
        if len(preserved_boxes) != 0 and len(preserved_labels) != 0:   
            boxes = preserved_boxes
            labels = preserved_labels
            img = crop_img

        return img, boxes, labels
    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, p={1}, th={2})'.format(self.size, self.p, self.th)

class RandomHorizontalFlip(object):
    """Horizontally flip the given image(numpy.ndarray) randomly with a given probability.
    Args:
        p (float): Probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, boxes, labels):
        """
        Args:
            img (numpy.ndarray): Image to be flipped.
            boxes: Bounding boxes of the image.
            labels: Labels of bounding boxes.
        Returns:
            img: Randomly flipped image.
            boxes: No operation.
            labels: No operation.
        """
        if random.random() < self.p:
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            for box in boxes:
                xmin = box[0]
                xmax = box[2]
                box[2] = w - xmin
                box[0] = w - xmax

        return img, boxes, labels

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class SubstractData(object):
    """Substract by mean.
    Args:
        mean (sequence): Mean values of selected dataset. The sequence is [B, G, R].
    """
    def __init__(self, mean):
        self.mean = mean
    def __call__(self, img, boxes, labels):
        """
        Args:
            img (numpy.ndarray): Image data to be shifted.
            boxes: Bounding boxes of the image.
            labels: Labels of bounding boxes.
        Returns:
            img: substracted image.
            boxes: No operation.
            labels: No operation.
        """
        img -= self.mean
        return img, boxes, labels

    def __repr__(self):
        return self.__class__.__name__ + '(mean={})'.format(self.mean)

class NormalizeBoundingBox(object):
    """Normalize bounding boxes.
    """
    def __call__(self, img, boxes, labels):
        """
        Args:
            img (numpy.ndarray): Image data for dim info.
            boxes: Bounding boxes of the image to be normalized.
            labels: Labels of bounding boxes.
        Returns:
            img: No operation.
            boxes: Normalized bounding boxes.
            labels: No operation.
        """
        h, w, _ = img.shape
        # normalize bounding boxes
        for box in boxes:
            box[0] = box[0]/float(w)
            box[2] = box[2]/float(w)
            box[1] = box[1]/float(h)
            box[3] = box[3]/float(h)
        return img, boxes, labels
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class Resize(object):
    """Resize the input image(numpy.ndarray) to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            then the sequence will be created.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        
    def __call__(self, img, boxes, labels):
        """
        Args:
            img (numpy.ndarray): Image to be resized.
            boxes: Bounding boxes of the image to be normalized.
            labels: Labels of bounding boxes.
        Returns:
            img: Resized image.
            boxes: No operation.
            labels: No operation.
        """
        img = cv2.resize(img, self.size).astype(np.float32)
        return img, boxes, labels
    def __repr__(self):
        return self.__class__.__name__ + '(size={})'.format(self.size)
