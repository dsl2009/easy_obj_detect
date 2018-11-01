from torchvision import transforms
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
def aug_color(igs):
    trans = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    ])
    ig = Image.fromarray(igs)
    ig = trans(ig)
    ig = np.asarray(ig)
    return ig