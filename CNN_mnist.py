import numpy as np
import mnist

def init(self,num_filters):
    self.num_filters = num_filters
    self.filters = np.random.randn
def iterate_regions(self, image):
    h, w = image.shape

    for i in range(h - 2):
      for j in range(w - 2):
        im_region = image[i:(i + 3), j:(j + 3)]
        yield im_region, i, j

def forward(self, input): 
    h,w=input.shape
    output = np.zeros((h - 2, w - 2, self.num_filters))

    for im_region, i, j in self.iterate_regions(input):
        output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

    return output
