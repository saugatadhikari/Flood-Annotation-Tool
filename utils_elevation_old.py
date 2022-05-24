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
from collections import defaultdict

import ipywidgets as widgets
from ipywidgets import *
import IPython.display as Disp



#######################################################################################################################
## Class For Interactive BBox Operations
#######################################################################################################################

class bbox_select():
    
    def __init__(self, ann_path: str, label_path: str, flood_class: int):
        """
            ann_path: path to the features file(.npy)
            label_path: path to the labels file(.npy)
            flood_class: 1 if flood, 0 if dry
        """
        
        ## Load annotation data
        self.data = np.load(ann_path)

        self.label_path = label_path
        self.flood_class = flood_class
        
        ## Extract color image from data
        self.color_image = self.data[:,:, :-1].astype('uint8')
        
        ## Extract elevaation data and convert to image
        self.elevation_map = self.data[:,:, -1].astype('float')
        
        # Normalize to 0 and 1
        self.elevation_map = ( self.elevation_map - np.min(self.elevation_map)) / \
                               (np.max(self.elevation_map) - np.min(self.elevation_map) )
        
        self.elevation_image = (self.elevation_map*255).astype("uint8")
        self.elevation_image = cv2.cvtColor(self.elevation_image, cv2.COLOR_BGR2RGB)
        
        
        ## Extract labels
        # initially there might be no labels
        try:
            self.labels = np.load(label_path)
        except FileNotFoundError:
            if flood_class:
                self.labels = np.zeros((self.elevation_map.shape[0], self.elevation_map.shape[1], 1))
            else:
                self.labels = np.ones((self.elevation_map.shape[0], self.elevation_map.shape[1], 1))
        
        ## Extract land indices
        land_idx = np.where(self.labels == 0) 
        
        ## Extract flood indices
        flood_idx = np.where(self.labels == 1)
        
        ## Create combined map
        if flood_class:
            self.combined_mask = np.zeros((self.labels.shape[0], self.labels.shape[1], 3))
        else:
            self.combined_mask = np.ones((self.labels.shape[0], self.labels.shape[1], 3))
        
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
        self.color_overlay_view = self.bbox_figure_ax2.imshow(self.color_image_overlay.copy(), cmap = 'gray')
        
        # Create fourth subplot
        self.bbox_figure_ax3 = self.bbox_figure.add_subplot(gs[2:, 2:])
        self.bbox_figure_ax3.set_title('4. Elevation Image Overlay')
        self.elevation_overlay_view = self.bbox_figure_ax3.imshow(self.elevation_image_overlay.copy(), cmap = 'gray')
        
        
    
    def cicle_img(self, img, pts):    
        img = cv2.circle(img, (int(pts[0]), int(pts[1])), radius = 2, color = (255, 0, 0), thickness=-1)
        return img

    
    def onclick(self, event):
        #display(str(event))
        self.selected_points = [event.xdata, event.ydata]
        
        self.bbox_figure
            
        self.image_view.set_data(self.cicle_img(self.color_image.copy(), self.selected_points))
        self.elevation_view.set_data(self.cicle_img(self.elevation_image.copy(), self.selected_points))
        self.color_overlay_view.set_data(self.cicle_img(self.color_image_overlay.copy(), self.selected_points))
        self.elevation_overlay_view.set_data(self.cicle_img(self.elevation_image_overlay.copy(), self.selected_points))

    
    # def bfs(self):
    #     # round selected point to nearest integer
    #     self.selected_point = (round(self.selected_points[0]), round(self.selected_points[1]))
    #     i, j = self.selected_point

    #     height, width = self.elevation_map.shape

    #     # get 8 neighboring pixels and their elevation
    #     # flooded_pixels = []
    #     bfs_queue = []

    #     bfs_queue.append(self.selected_point)

    #     bfs_visited = defaultdict(lambda: defaultdict(bool))
    #     bfs_visited[i][j] = True

    #     self.flood_labels = self.labels.copy()

    #     while bfs_queue:
    #         (i, j) = bfs_queue.pop(0)
    #         bfs_visited[i][j] = True

    #         # go through the 8 neighbors
    #         for l in [-1, 0, 1]:
    #             for r in [-1, 0, 1]:
    #                 if (l == r == 0):
    #                     continue

    #                 i_nei, j_nei = (i+l, j+r) # get the neighboring i and j
                    
    #                 # check for boundary cases
    #                 if i_nei < 0 or j_nei < 0 or i_nei >= width or j_nei >= height:
    #                     continue
                    
    #                 # check if already visited or not
    #                 if bfs_visited[i_nei][j_nei]:
    #                     continue
                    
    #                 # check current pixel's elevation with neighbor's elevation
    #                 if self.flood_class:
    #                     if (self.elevation_map[i_nei][j_nei] <= self.elevation_map[i][j]) and (self.flood_labels[j_nei][i_nei] == 0):
    #                         self.flood_labels[j_nei][i_nei] = 1
    #                         bfs_queue.append((i_nei, j_nei))
    #                 else:
    #                     if (self.elevation_map[i_nei][j_nei] >= self.elevation_map[i][j]) and (self.flood_labels[j_nei][i_nei] == 1):
    #                         self.flood_labels[j_nei][i_nei] = 0
    #                         bfs_queue.append((i_nei, j_nei))
    
    def bfs(self):
        # round selected point to nearest integer
        self.selected_point = (round(self.selected_points[0]), round(self.selected_points[1]))
        i, j = self.selected_point

        height, width = self.elevation_map.shape

        # get 8 neighboring pixels and their elevation
        # flooded_pixels = []
        bfs_queue = []

        # bfs_queue.append(self.selected_point)
        bfs_queue.append((j,i))

        bfs_visited = defaultdict(lambda: defaultdict(bool))
        bfs_visited[j][i] = True

        self.flood_labels = self.labels.copy()

        while bfs_queue:
            (j, i) = bfs_queue.pop(0)
            bfs_visited[j][i] = True

            # go through the 8 neighbors
            for l in [-1, 0, 1]:
                for r in [-1, 0, 1]:
                    if (l == r == 0):
                        continue

                    j_nei, i_nei = (j+l, i+r) # get the neighboring i and j
                    
                    # check for boundary cases
                    if i_nei < 0 or j_nei < 0 or i_nei >= width or j_nei >= height:
                        continue
                    
                    # check if already visited or not
                    if bfs_visited[j_nei][i_nei]:
                        continue
                    
                    # check current pixel's elevation with neighbor's elevation
                    if self.flood_class:
                        if (self.elevation_map[j_nei][i_nei] <= self.elevation_map[j][i]) and (self.flood_labels[j_nei][i_nei] == 0):
                            self.flood_labels[j_nei][i_nei] = 1
                            bfs_queue.append((j_nei, i_nei))
                    else:
                        if (self.elevation_map[j_nei][i_nei] >= self.elevation_map[j][i]) and (self.flood_labels[j_nei][i_nei] == 1):
                            self.flood_labels[j_nei][i_nei] = 0
                            bfs_queue.append((j_nei, i_nei))


    def plot_bfs_result(self):
        print(self.flood_labels.shape)

        ## Extract land indices
        land_idx = np.where(self.flood_labels == 0) 

        ## Extract flood indices
        flood_idx = np.where(self.flood_labels == 1)

        ## Create combined map
        if self.flood_class:
            combined_mask = np.zeros((self.flood_labels.shape[0], self.flood_labels.shape[1], 3))
        else:
            combined_mask = np.ones((self.flood_labels.shape[0], self.flood_labels.shape[1], 3))

        combined_mask[land_idx[0], land_idx[1], 2] = 255
        combined_mask[flood_idx[0], flood_idx[1], 0] = 255
        combined_mask = combined_mask.astype("uint8")

        ## Overlay color mask        
        color_image_overlay = cv2.addWeighted(self.color_image.copy(), 0.7, combined_mask.copy(), 0.3, 0.0)
        elevation_image_overlay = cv2.addWeighted(self.elevation_image.copy(), 0.7, combined_mask.copy(), 0.3, 0.0)

        # Create new figure
        bbox_figure = plt.figure(1, constrained_layout=True, figsize=(9, 9))

        # Create a grid spec for figure
        gs = bbox_figure.add_gridspec(2, 2)

        # Create third subplot
        bbox_figure_ax2 = bbox_figure.add_subplot(gs[0:1, 0:1])
        bbox_figure_ax2.set_title('3. Color Image Overlay')
        bbox_figure_ax2.imshow(color_image_overlay.copy(), cmap = 'gray')

        # Create fourth subplot
        bbox_figure_ax3 = bbox_figure.add_subplot(gs[0:1, 1:2])
        bbox_figure_ax3.set_title('4. Elevation Image Overlay')
        bbox_figure_ax3.imshow(elevation_image_overlay.copy(), cmap = 'gray')

    
    def commit_result(self):
        np.save(self.label_path, self.flood_labels)
    

    def plot_trisurface(self):
        X = []
        Y = []
        Z = []

        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                X.append(i)
                Y.append(j)
                Z.append(self.data[i][j][-1])

        plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')

