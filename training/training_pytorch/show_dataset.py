from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
path1 = "./testdataset/0.npz"
data = np.load(path1)
data = data['arr_0']
print(data.shape)
plt.imshow(data[0],cmap='Accent')
plt.show()
plt.imshow(data[79],cmap='Accent')
plt.show()
plt.imshow(data[79])
plt.show()


