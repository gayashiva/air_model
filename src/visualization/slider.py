import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

# generate a five layer data
data = np.random.randint(10, size=(5, 5, 5))
# current layer index start with the first layer
idx = 0

# figure axis setup
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.15)

# display initial image
im_h = ax.imshow(data[:, :, idx], cmap='hot', interpolation='nearest')

# setup a slider axis and the Slider
ax_depth = plt.axes([0.23, 0.02, 0.56, 0.04])
slider_depth = Slider(ax_depth, 'depth', 0, data.shape[2]-1, valinit=idx)

# update the figure with a change on the slider
def update_depth(val):
    idx = int(round(slider_depth.val))
    im_h.set_data(data[:, :, idx])

slider_depth.on_changed(update_depth)

plt.show()