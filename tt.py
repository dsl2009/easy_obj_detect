from skimage import io
from dsl_data import utils
from matplotlib import pyplot as plt
ig = io.imread('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/bdd100k/images/10k/test/ac6d4f42-00000000.jpg')
image, window, scale, padding, crop = utils.resize_image_fixed_size(ig, image_size=[768, 1280])
print(window, scale, padding, crop)
plt.imshow(image)
plt.show()