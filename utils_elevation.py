import numpy as np
import json

import cv2
from PIL import Image
#import skimage
#from skimage.draw import disk

import matplotlib
from matplotlib import pyplot as plt

import re
import os

import ipywidgets as widgets
from ipywidgets import *
import IPython.display as Disp



#######################################################################################################################
## Class For Interactive BBox Operations
#######################################################################################################################

class bbox_select():
    
    def __init__(self, ann_path, label_path):
        
        ## Load annotation data
        data = np.load(ann_path)
        
        ## Extract color image from data
        self.color_image = data[:,:, :-1].astype('uint8')
        
        ## Extract elevaation data and convert to image
        self.elevation_map = data[:,:, -1].astype('float')
        
        # Normalize to 0 and 1
        self.elevation_map = ( self.elevation_map - np.min(self.elevation_map)) / \
                               (np.max(self.elevation_map) - np.min(self.elevation_map) )
        
        self.elevation_image = (self.elevation_map*255).astype("uint8")
        self.elevation_image = cv2.cvtColor(self.elevation_image, cv2.COLOR_BGR2RGB)
        
        
        ## Extract labels
        labels = np.load(label_path)
        
        ## Extract land indices
        land_idx = np.where(labels == 0) 
        
        ## Extract flood indices
        flood_idx = np.where(labels == 1)
        
        ## Create combined map
        self.combined_mask = np.zeros((labels.shape[0], labels.shape[0], 3))
        
        self.combined_mask[land_idx[0], land_idx[1], 2] = 255
        self.combined_mask[flood_idx[0], flood_idx[1], 0] = 255
        self.combined_mask = self.combined_mask.astype("uint8")

        ## Overlay color mask        
        self.color_image_overlay = cv2.addWeighted(self.color_image.copy(), 0.7, self.combined_mask.copy(), 0.3, 0.0)
        self.elevation_image_overlay = cv2.addWeighted(self.elevation_image.copy(), 0.7, self.combined_mask.copy(), 0.3, 0.0)
        
        
        # Array to store polygon coordinates
        self.selected_points = []
        
        # Create new figure
        self.bbox_figure = plt.figure(1, constrained_layout=True, figsize=(9, 9))
        
        # Create event handler for clicks on matplotlib canvas
        self.mouse_event = self.bbox_figure.canvas.mpl_connect('button_press_event', self.onclick)
        
        # Create a grid spec for figure
        gs = self.bbox_figure.add_gridspec(4, 4)
        
        # Create first subplot
        self.bbox_figure_ax1 = self.bbox_figure.add_subplot(gs[0:2, 0:2])
        self.bbox_figure_ax1.set_title('1. Color Image')
        self.image_view = self.bbox_figure_ax1.imshow(self.color_image.copy())
        
        # Create Second subplot
        self.bbox_figure_ax4 = self.bbox_figure.add_subplot(gs[0:2, 2:])
        self.bbox_figure_ax4.set_title('2. Elevation Image')
        self.elevation_view = self.bbox_figure_ax4.imshow(self.elevation_image.copy())
        
        
        # Create third subplot
        self.bbox_figure_ax2 = self.bbox_figure.add_subplot(gs[2:, 0:2])
        self.bbox_figure_ax2.set_title('3. Color Image Overlay')
        self.bbox_figure_ax2.imshow(self.color_image_overlay.copy(), cmap = 'gray')
        
        # Create fourth subplot
        self.bbox_figure_ax3 = self.bbox_figure.add_subplot(gs[2:, 2:])
        self.bbox_figure_ax3.set_title('4. Elevation Image Overlay')
        self.bbox_figure_ax3.imshow(self.elevation_image_overlay.copy(), cmap = 'gray')
        
        
    
    def cicle_img(self, img, pts):    
        img = cv2.circle(img, (int(pts[0]), int(pts[1])), radius = 2, color = (255, 0, 0), thickness=-1)
        return img

    
    def onclick(self, event):
        #display(str(event))
        self.selected_points = [event.xdata, event.ydata]
        
        self.bbox_figure
            
        self.image_view.set_data(self.cicle_img(self.color_image.copy(), self.selected_points))
        self.elevation_view.set_data(self.cicle_img(self.elevation_image.copy(), self.selected_points))