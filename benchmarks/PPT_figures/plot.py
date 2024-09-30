# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc('ytick', labelsize=20)

species = ("ResNeXt-50", "BERT")
penguin_means = {
    'MF': (5.25, 11.69),
    'TS': (108.83, 790.28),
    'Sampling': (1.14, 2.02),
}


x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained', figsize=(8,4))

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3, weight='bold', fontsize=20)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_yscale('log')
ax.set_ylabel('Search Time (seconds)', weight='bold', fontsize=20)
# ax.set_title('Penguin attributes by species')
ax.set_xticks(x + width, species, weight='bold', fontsize=20)
# ax.legend(loc='upper left', ncols=3)
# ax.set_ylim(0, 250)

ax.grid()
fig.savefig(f"./OCGGS.png")
plt.show()







# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc('ytick', labelsize=20)

species = ("MobileNetV2", "1D-IR", "ResNet50")
penguin_means = {
    'ST': (1/3.1, 1/2.3),
    'Ansor': (1, 1),
    'ETO': (1/5.9, 1/3.1),
}


x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained', figsize=(8,4))

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3, weight='bold', fontsize=20)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_yscale('log')
ax.set_ylabel('Search Time (seconds)', weight='bold', fontsize=20)
# ax.set_title('Penguin attributes by species')
ax.set_xticks(x + width, species, weight='bold', fontsize=20)
# ax.legend(loc='upper left', ncols=3)
# ax.set_ylim(0, 250)

ax.grid()
fig.savefig(f"./OCGGS.png")
plt.show()
