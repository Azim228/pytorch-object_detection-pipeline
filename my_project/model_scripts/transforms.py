import random
import torch

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            if "boxes" in target:
                bbox = target["boxes"]
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]] 
                target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = F.vflip(image) 
            if "boxes" in target:
                bbox = target["boxes"]
                bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
                target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = F.vflip(target["masks"])
            # if "keypoints" in target:
            #     keypoints = target["keypoints"]
            #     keypoints = _flip_coco_person_keypoints(keypoints, width)
            #     target["keypoints"] = keypoints
        return image, target


class Invert(object):
    def __init__(self):
        pass

    def __call__(self, image, target):
        image = -image + 1

        return image, target


class AddNoise(object):
    def __init__(self, prob=0.5, intensity=0.05):
        self.prob = prob
        self.intensity = intensity

    def __call__(self, image, target):
        if random.random() < self.prob:
            channels, height, width = image.shape
            noise = torch.rand((channels, height, width), device=image.device)*self.intensity #add random noise of max 13 graylevels (0.05*255=13)
            image = image + noise
            image[image>1]=1
            image[image<0]=0

            # if "keypoints" in target:
            #     keypoints = target["keypoints"]
            #     keypoints = _flip_coco_person_keypoints(keypoints, width)
            #     target["keypoints"] = keypoints
        return image, target

class ChangeIntensity(object):
    """
    Increase/Decrease brightness and contrast by random number from interval [-param,param]
    
    image: Tensor in range [0...1]
    
    brightness: float in range [0...1] additive factor to pixel values. 
    brightness = 0 would be neutral
    
    contrast float in range [0...1] scaling factor to all pixel values. pixels
    are scaled by 1+ rdm[-contrast,contrast]. contrast=0 would lead to a scaling
    by 1+0=1, which would be neutral.
    """
    def __init__(self, prob=0.5, brightness=0.1, contrast=0.1):
        self.prob = prob
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image, target):
        if random.random() < self.prob:
            
            scale_factor = 1 + torch.FloatTensor(1).uniform_(-self.contrast,self.contrast).to(image.device)
            image = image * scale_factor
            
            add_factor = torch.FloatTensor(1).uniform_(-self.brightness,self.brightness).to(image.device)
            image = image + add_factor
            
            image[image>1]=1
            image[image<0]=0

            # if "keypoints" in target:
            #     keypoints = target["keypoints"]
            #     keypoints = _flip_coco_person_keypoints(keypoints, width)
            #     target["keypoints"] = keypoints
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
    



