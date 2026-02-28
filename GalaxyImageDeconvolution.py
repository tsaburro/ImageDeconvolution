# Project Title: Galaxy Image Deconvolution
# Start Date: 2/20/2026

# Section 0: Tools and imports
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.signal import wiener
from skimage import restoration
from scipy.signal import convolve2d as conv2

# Section 1: Pulling data from NASA FITS database
# Load image from FITS
dataset = r"C:\Users\tybos\OneDrive\main\Research and Projects\Astro\Projects\Galaxy Deconvolution\502nmos.fits"
# ^ using r will turn the file into a raw string allowing you to actually find it in the project folder

with fits.open(dataset) as hdul:
    original_image = hdul[0].data
# ^ better to open data with "with" because it auto closes the data, preventing future errors

# Getting the image to look better, focusing on percentiles of brightest pixels to get cleanest image quality
p_low, p_high = np.percentile(original_image, (33.3, 99.9))
scaled = np.clip((original_image - p_low) / (p_high - p_low), 0, 1)

# Normalize the dataset (overwritting original image with new scaled image)
image = scaled / np.max(scaled)

"""
# Show the normalized data
plt.figure(figsize= (12,8))
plt.imshow(image, cmap = "inferno", vmin = 0, vmax = 1)
plt.colorbar()
plt.title("Original image")
plt.show()
"""

# Section 2: Create a telescope blur / using guassian kernel
sigma = 15 # blur strength
blurred = gaussian_filter(image, sigma = sigma)

"""
plt.figure(figsize= (12,8))
plt.imshow(blurred, cmap = "inferno", vmin = 0, vmax = 1)
plt.colorbar()
plt.title("Blurred image")
plt.show()
"""

# Section 3: Add noise to blurred image
noise_level = 0.2
noisy = blurred + noise_level * np.random.normal(size = image.shape)

"""
plt.figure(figsize= (12,8))
plt.imshow(noisy, cmap = "inferno", vmin = 0, vmax = 1)
plt.colorbar()
plt.title("Noisy image")
plt.show()
"""

# ===========================
#  REDO RECONSTRUCTION LATER
# ===========================

# Section 4: Completeing naive deconvolution using wiener filter to reconstruct the noisy image
# reconstructed = wiener(noisy, (2, 2))
# reconstructed = np.clip(reconstructed, 0, 1)

# Resolving the doconvolution issue using PSF (point spread function) instead of wiener
psf = np.ones((5,5)) / 25

# new noisy image to work with new PSF algorithm deconvolution
noisy_PSF = conv2(image, psf, 'same')

# new deconvolution algorithm in action
reconstructed_PSF = restoration.richardson_lucy(noisy_PSF, psf, num_iter = 30)

"""
plt.figure(figsize= (12,8))
plt.imshow(reconstructed, cmap = "inferno", vmin = 0, vmax = 1)
plt.colorbar()
plt.title("Reconstruction of noisy image with wiener filter")
plt.show()
"""

# Section 5: Combining all images into one plot for ease of visual affect
fig, axes = plt.subplots(1, 4, figsize = (18,6))
titles = ["Original", "Blur", "Noise", "Reconstructed"]
images = [image, blurred, noisy_PSF, reconstructed_PSF]

for ax, img, title in zip(axes, images, titles):
    im = ax.imshow(img, cmap = 'jet', vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis('off')  # hides axes ticks for cleaner look

# Add a single colorbar for all images
cbar_ax = fig.add_axes([0.1, 0.08, 0.8, 0.03])
cbar = fig.colorbar(im, cax = cbar_ax, orientation = "horizontal", fraction = 0.05, pad = 0.025)
cbar.set_label("Scaled Intensity")

plt.tight_layout()
plt.show()