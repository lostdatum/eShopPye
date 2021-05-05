# This file contains functions used in the data_exploration.ipynb and
# data_viz_image.ipynb files.


import os
import cv2
import matplotlib.pyplot as plt
import textwrap
import numpy as np
from typing import Union


def load_id(path: str, imageids: Union[str, list]):
    """Searches and loads images identified their 'imageid' in path.

    Parameters
    ----------

    path : str
        Path to folder containing images.
            
    imageids : str, list of str
        List of 'imageid' strings of images.


    Returns
    -------

    images : list of numpy.ndarray
        Images encoded as described in the docs of cv2.imread().
    """
    
    filenames = os.listdir(path)
    
    if isinstance(imageids, str): # single image
    
        img_id = imageids
        
        # Search pattern in filename
        pattern = 'image_{}_'.format(img_id)
        matches = [name for name in filenames if pattern in name]
        
        # Check results
        if len(matches) < 1:
            raise FileNotFoundError("No image matching {} was found.".format(pattern))
        elif len(matches) > 1:
            raise FileNotFoundError("Multiple images matching {} were found: \n{}".format(pattern, matches))
        else:
            img_name = matches[0]
        
        # Load & save
        img = cv2.imread(os.path.join(path, img_name))
        
        return img
    
    else: # multiple images
    
        images = list()
        
        # Iterate through IDs
        for img_id in imageids:
            
            # Search pattern in filename
            pattern = 'image_{}_'.format(img_id)
            matches = [name for name in filenames if pattern in name]
            
            # Check results
            if len(matches) < 1:
                raise FileNotFoundError("No image matching {} was found.".format(pattern))
            elif len(matches) > 1:
                raise FileNotFoundError("Multiple images matching {} were found: \n{}".format(pattern, matches))
            else:
                img_name = matches[0]
            
            # Load & save
            img = cv2.imread(os.path.join(path, img_name))
            images.append(img)
        
    return images



def disp_grid(images: list, rows: int, cols: int, channel: int=None, cmap: str=None, titles: list=None, suptitle: str=None):
    """Displays images in a grid fashion.

    Parameters
    ----------

    images : list of numpy.ndarray
        Images encoded as numpy.ndarray(n_pixels_x, n_pixels_y, n_channels).
            
    rows : int
        Number of rows. Number of rows times number columns must be higher than length of images list.

    cols : int
        Number of rows. Number of rows times number columns must be higher than length of images list.
        
    channel : int, optional
        Color channel to select.
    
    cmap : str, optional
        Colormap to be used, for images whith only one color channel.
        When images have multiple colors channels, the channel argument must be supplied.
    
    titles : list of str, optional
        Titles for images. Length must match images list length.
        
    suptitle : str, optional
        Figure title.


    Returns
    -------

    None
    """
    
    if rows * cols < len(images):
        raise ValueError("Grid size is too small to plot all images.")
    
    if titles is not None and len(titles) != len(images):
        raise ValueError("Length of titles and images do not match.")
        
    if channel is not None and channel > images[0].shape[2]-1:
        raise ValueError("Sypplied color channel is out of bounds.")
        
    if cmap is not None and images[0].shape[2] > 1 and channel is None:
        raise ValueError("Custom colormap cannot be used with more than one color channel. Supply channel to be used.")

    fig = plt.figure(figsize=(16,9))
    
    for i in range(len(images)):
        
        ax = fig.add_subplot(rows, cols, i+1)
        ax.axis('off')
        
        if cmap is not None:
            if channel is not None:
                ax.imshow(images[i][:, :, channel], cmap=cmap)
            else:
                ax.imshow(images[i], cmap=cmap)
        else:
            ax.imshow(images[i])
            
        if titles is not None:
            title = textwrap.fill(titles[i], width=40, break_long_words=False)
            ax.set_title(title, fontsize=10)
            
        if suptitle:
            fig.suptitle(suptitle, fontsize=16)



def mean_img(images: list):
    """Computes mean of images.

    Parameters
    ----------

    images : list of numpy.ndarray
        Images encoded as numpy.ndarray(n_pixels_x, n_pixels_y, n_channels).
            
    Returns
    -------

    images_mean : numpy.ndarray
        Mean of images. Image encoded as numpy.ndarray(n_pixels_x, n_pixels_y, n_channels).
    """
    
    images = np.asarray(images)
    images_mean = images.mean(axis=0)
    images_mean = images_mean.astype('int')
    
    return images_mean
    
    
def var_img(images: list, images_mean: np.ndarray=None):
    """Computes mean of images.

    Parameters
    ----------

    images : list of numpy.ndarray
        Images encoded as numpy.ndarray(n_pixels_x, n_pixels_y, n_channels).
        
    images_mean : numpy.ndarray, optional
        If provided, will be used instead of computing mean of images.
        Encoded as numpy.ndarray(n_pixels_x, n_pixels_y, n_channels).
            
    Returns
    -------

    images_var : numpy.ndarray
        Variance of images. Image encoded as numpy.ndarray(n_pixels_x, n_pixels_y, n_channels).
    """
    
    if images_mean is None:
        images_mean = mean_img(images)
    
    images_var = np.zeros((images[0].shape[0], images[0].shape[1], images[0].shape[2]))
    
    for i in range(len(images)):
        images_var += (((images[i] - images_mean)/255)**2)*255
    
    images_var /= len(images)
    images_var= images_var.astype('int')
    
    return images_var
    
    
    
def grayscale(images: list):
	"""Converts images in list to grayscale.

	Parameters
	----------

	images : list of numpy.ndarray
		Images encoded as numpy.ndarray(n_pixels_x, n_pixels_y, n_channels).
			
	Returns
	-------

	images_gs: list of numpy.ndarray
		List of grayscale images encoded as numpy.ndarray(n_pixels_x, n_pixels_y, 1).
	"""
	
	images_gs = []

	for i in range(len(images)):
		images_gs.append(np.expand_dims(images[i].mean(axis=2), axis=2))
	
	return images_gs
    