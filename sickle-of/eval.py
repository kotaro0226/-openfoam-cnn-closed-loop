import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from args import args
from matplotlib.colors import LinearSegmentedColormap

# Load data
data = np.load('snapshots/subsampled.npz')
x, y = data['x'], data['y']

data = np.load(os.path.join(args.output_dir, 'test_maxent.npz'))
X_test, Y_test, X_train, Y_train = (
    torch.tensor(data['X_test'], dtype=torch.float32),
    torch.tensor(data['Y_test'], dtype=torch.float32),
    torch.tensor(data['X_train'], dtype=torch.float32),
    torch.tensor(data['Y_train'], dtype=torch.float32),
)

# Load the saved model
model = torch.load(f"models/{args.arch}/1")
model.eval()

# Ensure the model has been loaded correctly
print(model)

print(X_test.shape, Y_test.shape, X_train.shape, Y_train.shape)

errors = []
with torch.no_grad():
    for i in range(X_test.shape[0]):
        ypred = model(X_test)
        error = torch.abs(ypred - Y_test) / ypred.shape[0]
        errors.append(error.numpy())
    errors = np.array(errors)

# Define a green-yellow-red colormap
cmap_colors = [(0, "green"), (0.5, "yellow"), (1, "red")]
cmap_gyr = LinearSegmentedColormap.from_list("GreenYellowRed", cmap_colors)

#emin = np.min(errors)
#emax = np.max(errors)
emin = 0
emax = 0.015

for t in range(X_test.shape[0]):
    plt.clf()
    plt.figure(figsize=(9, 2))
    plt.scatter(x, y, c=errors[t, :], marker='.', cmap=cmap_gyr, vmin=emin, vmax=emax)
    plt.xlim([-25, 65])
    plt.ylim([-10, 10])
    plt.axis('equal') 
    cbar = plt.colorbar(ticks=np.linspace(emin, emax, 5))
    cbar.set_label(r'L1 error')
    plt.savefig(os.path.join(args.plot_dir, f'errors_{t:04d}_random.png'), dpi=100)
