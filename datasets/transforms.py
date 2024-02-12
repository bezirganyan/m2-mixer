from random import random

import torchvision.transforms as T

class RuinModality:
    """
    Ruin a modality with a given probability.
    Args:
        modality (str): Modality to ruin. Can be 'image', 'text', 'both' or 'xor'.
        p (float): Probability of ruining a modality.
    """
    def __init__(self, modality='xor', p=0.5):
        assert modality in ['image', 'text', 'both', 'xor']
        self.modality = modality
        self.p = p
        self.blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))

    def blur_image(self, image):
        return self.blur(image)

    def remove_text(self, text):
        words = text.split()
        rtext = ' '.join([word for word in words if random() > self.p])
        if rtext == '':
            return words[0]
        return rtext

    def __call__(self, sample):
        if self.modality == 'image':
            sample['image'] = self.blur_image(sample['image'])
        elif self.modality == 'text':
            sample['text'] = self.remove_text(sample['text'])
        elif self.modality == 'both':
            sample['image'] = self.blur_image(sample['image'])
            sample['text'] = self.remove_text(sample['text'])
        elif self.modality == 'xor':
            if random() > 0.5:
                sample['image'] = self.blur_image(sample['image'])
            else:
                sample['text'] = self.remove_text(sample['text'])
        return sample

