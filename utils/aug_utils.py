from torchvision import transforms
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

trans = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 ),
        ])

for x in range(10):
    ig = Image.open('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/r2testb/412.jpg')
    ig = trans(ig)
    ig = np.asarray(ig)
    plt.imshow(ig)
    plt.show()