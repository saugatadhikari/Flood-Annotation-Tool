from collections import defaultdict

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri

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
            self.labels = np.full((self.elevation_map.shape[0], self.elevation_map.shape[1], 1), -1)
            # if flood_class:
            #     self.labels = np.zeros((self.elevation_map.shape[0], self.elevation_map.shape[1], 1))
            # else:
            #     self.labels = np.ones((self.elevation_map.shape[0], self.elevation_map.shape[1], 1))
        
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

        # Create a refresh button to display on matplotlib canvas
        polygon_button = widgets.Button(description='Draw Polygon',
                                        disabled=False,
                                        button_style = 'danger', # 'success', 'info', 'warning', 'danger' or ''
                                        tooltip='Click me',
                                        icon='polygon' # (FontAwesome names without the `fa-` prefix)
                                       )
        # Display Refresh Button
        Disp.display(polygon_button)
        polygon_button.on_click(self.polygon)
        
        
        # Create a preview button to display on matplotlib canvas
        point_button = widgets.Button(description='Select Points',
                                        disabled=False,
                                        button_style = 'info', # 'success', 'info', 'warning', 'danger' or ''
                                        tooltip='Click me',
                                        icon='point' # (FontAwesome names without the `fa-` prefix)
                                       )
        # Display Refresh Button
        Disp.display(point_button)
        point_button.on_click(self.point)
        
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

        self.annotation_type = "point" # we set point annotation by default

        
        
    
    def polygon(self, _):
        self.annotation_type = "polygon"

    def point(self, _):
        self.annotation_type = "point"
    
    def circle_img(self, img, pts):
        c_color = (255, 0, 0)
        if self.flood_class == 0:
            c_color = (0, 0, 255)
        for pt in pts: 
            img = cv2.circle(img, (int(pt[0]), int(pt[1])), radius = 2, color = c_color, thickness=-1)
        return img

    def poly_img(self, img, pts):
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        c_color = (255, 0, 0)
        if self.flood_class == 0:
            c_color = (0, 0, 255)
        cv2.polylines(img, [pts], True, c_color, 1)
        return img

    
    def onclick(self, event):
        selected_point = [event.xdata, event.ydata]
        self.selected_points.append(selected_point)

        if self.annotation_type == "point":
            self.bbox_figure

            self.image_view.set_data(self.circle_img(self.color_image.copy(), self.selected_points))
            self.elevation_view.set_data(self.circle_img(self.elevation_image.copy(), self.selected_points))
            self.color_overlay_view.set_data(self.circle_img(self.color_image_overlay.copy(), self.selected_points))
            self.elevation_overlay_view.set_data(self.circle_img(self.elevation_image_overlay.copy(), self.selected_points))

        elif self.annotation_type == "polygon":
            if len(self.selected_points)>1:
                self.bbox_figure
                self.image_view.set_data(self.poly_img(self.color_image.copy(), self.selected_points))
                self.elevation_view.set_data(self.poly_img(self.elevation_image.copy(), self.selected_points))
                self.color_overlay_view.set_data(self.poly_img(self.color_image_overlay.copy(), self.selected_points))
                self.elevation_overlay_view.set_data(self.poly_img(self.elevation_image_overlay.copy(), self.selected_points))
    
    def bfs(self):
        if self.annotation_type == "polygon":
            if len(self.selected_points) > 1:
                # Convert Coordiantes of the polygon to numpy array
                self.np_selected_points = np.array([self.selected_points], 'int')

                #Get polygon region
                self.fill_mask = cv2.fillPoly(np.zeros(self.elevation_map.shape, np.uint8), self.np_selected_points, [1])
                polygon_area = np.where(self.fill_mask != 0)
                self.selected_points = list(zip(polygon_area[1], polygon_area[0]))

        height, width = self.elevation_map.shape

        # get 8 neighboring pixels and their elevation
        bfs_queue = []
        bfs_visited = defaultdict(lambda: defaultdict(bool))

        self.flood_labels = self.labels.copy()
        
        for selected_point in self.selected_points:
            # round selected point to nearest integer
            try:
                selected_point = (round(selected_point[0]), round(selected_point[1]))
            except TypeError:
                continue
            i, j = selected_point

            # this pixel might be selected twice
            if bfs_visited[j][i]:
                continue

            if self.flood_class:
                self.flood_labels[j][i] = 1
            else:
                self.flood_labels[j][i] = 0

            bfs_queue.append((j,i))

            bfs_visited[j][i] = True

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
                            if (self.elevation_map[j_nei][i_nei] <= self.elevation_map[j][i]) and (self.flood_labels[j_nei][i_nei] != 1):
                                self.flood_labels[j_nei][i_nei] = 1
                                bfs_queue.append((j_nei, i_nei))
                        else:
                            if (self.elevation_map[j_nei][i_nei] >= self.elevation_map[j][i]) and (self.flood_labels[j_nei][i_nei] != 0):
                                self.flood_labels[j_nei][i_nei] = 0
                                bfs_queue.append((j_nei, i_nei))


    def plot_bfs_result(self):

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
        gs = bbox_figure.add_gridspec(1, 2)

        # Create third subplot
        bbox_figure_ax2 = bbox_figure.add_subplot(gs[0, 0])
        bbox_figure_ax2.set_title('3. Color Image Overlay')
        bbox_figure_ax2.imshow(color_image_overlay.copy(), cmap = 'gray')

        # Create fourth subplot
        bbox_figure_ax3 = bbox_figure.add_subplot(gs[0, 1])
        bbox_figure_ax3.set_title('4. Elevation Image Overlay')
        bbox_figure_ax3.imshow(elevation_image_overlay.copy(), cmap = 'gray')

    
    def commit_result(self):
        np.save(self.label_path, self.flood_labels)
    

    def plot_3d_old_2(self):
        land_idx = np.where(self.flood_labels == 0)
        flood_idx = np.where(self.flood_labels == 1)

        cm = np.zeros((224, 224,3))

        cm[land_idx[0], land_idx[1], 0] = 255
        cm[flood_idx[0], flood_idx[1], 2] = 255
        cm = cm.astype("uint8")

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
        ax.scatter(X, Y, Z, edgecolor='none',c=cm[:,:,-1], cmap='jet')


    def plot_3d_old(self):
        land_idxs = np.where(self.flood_labels == 0)
        flood_idxs = np.where(self.flood_labels == 1)
        unk_idxs = np.where(self.flood_labels == -1)

        # cm = np.zeros((self.flood_labels.shape[0], self.flood_labels.shape[1]))
        cm = np.full((self.flood_labels.shape[0], self.flood_labels.shape[1]), 135)

        cm[land_idxs[0], land_idxs[1]] = 0
        cm[flood_idxs[0], flood_idxs[1]] = 255
        cm[unk_idxs[0], unk_idxs[1]] = 135

        cm = cm.astype("uint8")

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

        ax.plot_trisurf(X, Y, Z, edgecolor='none', color='green')
        ax.scatter(X, Y, Z, edgecolor='none',c=cm, cmap='jet')

    
    def plot_3d(self):

        X0 = []
        Y0 = []
        Z0 = []

        X1 = []
        Y1 = []
        Z1 = []

        X2 = []
        Y2 = []
        Z2 = []

        X = []
        Y = []
        Z = []

        for i in range(self.flood_labels.shape[0]):
            for j in range(self.flood_labels.shape[1]):
                X.append(i)
                Y.append(j)
                Z.append(self.data[i][j][-1])

                if self.flood_labels[i][j] == 0:
                    X0.append(i)
                    Y0.append(j)
                    Z0.append(self.data[i][j][-1])
                elif self.flood_labels[i][j] == 1:
                    X1.append(i)
                    Y1.append(j)
                    Z1.append(self.data[i][j][-1])
                else:
                    X2.append(i)
                    Y2.append(j)
                    Z2.append(self.data[i][j][-1])

        plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        
        # ax.plot_trisurf(X, Y, Z, edgecolor='none', color='green')
        
        ax.plot_trisurf(X, Y, Z, edgecolor='none', color='green', alpha=0.5)
        # ax.scatter(X2, Y2, Z2, c='green')

        ax.scatter(X0, Y0, Z0, c='blue')
        ax.scatter(X1, Y1, Z1, c='red')   
        

        
        

