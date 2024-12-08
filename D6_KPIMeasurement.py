import tkinter as tk
import torch
import math
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog, scrolledtext
import os
import sys
from tkinter import ttk
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics import SAM
import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


######## Below are for video-level analysis

def video_analysis():
    # Create a new Toplevel window
    new_window = tk.Toplevel(root)
    new_window.title("Video-level analysis")
    new_window.geometry("500x160")  # Adjusted the size of the window

    # Multiline text for the label
    text = """This is a secondary interface for videos-level Gas Flares analysis. 
    More functions are under-development."""

    # Add a label with multiline text to the new window
    label = tk.Label(new_window, text=text, justify=tk.CENTER)
    label.pack(padx=10, pady=20)
## 做了两个更改： 将flare smoke masks合并， 得到可视化的contour orientation
###########################################################
###########################################
def open_results_visualizer_window():  # THIS IS THE GROUND TRUTH
    # Paths of folders
    folder_path = entry_dir1.get()
    output_folder = entry_dir2.get()

    detection_images_path = os.path.join(output_folder, "Detection_Images")
    segmentation_visualized_images_path = os.path.join(output_folder, "Segmentation_Visualized_flame_smoke")
    flame_orientation_images_path = os.path.join(output_folder, "Segmentation_Visualized_flame_orientation")
    #  add sub-window
    segmentation_visualized_path1 = os.path.join(output_folder, "Segmentation_Visualized_flame")
    # this is the imperfect mask
    segmentation_visualized_path2 = os.path.join(output_folder, "Segmentation_Visualized_smoke")
    ######################
    value_path1 = os.path.join(output_folder, "flame_size")
    value_path2 = os.path.join(output_folder, "smoke_size")
    value_path3 = os.path.join(output_folder, "flame_smoke_ratio")
    value_path4 = os.path.join(output_folder, "flame_orientation")
    # Function to get the first image from a folder

    def create_black_image(width, height):  # 定义无检测情况
        # This function creates a black image of the specified width and height
        black_image = Image.new('RGB', (width, height), (0, 0, 0))
        return ImageTk.PhotoImage(black_image)


    def get_first_image(folder):
        for file in os.listdir(folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp','.txt')):
                return file
        return None  # No image found

    def find_matching_image(folder, partial_filename):
        if not partial_filename:
            return None
        for file in os.listdir(folder):
            if partial_filename.lower() in file.lower() and file.lower().endswith(
                    ('.png', '.jpg', '.jpeg', '.gif', '.bmp','.txt')):
                return os.path.join(folder, file)
        return None

    # Function to get a list of all image files in a folder
    def get_all_images(folder):
        return [file for file in os.listdir(folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Function to update the displayed images
    def update_images(index):

        # Update original image
        original_image_path = os.path.join(folder_path, all_images[index])
        original_image = load_image(original_image_path, max_size=(400, 300))
        canvas1.itemconfig(image_on_canvas1, image=original_image)

        # Update detection image
        detection_image_path = os.path.join(detection_images_path, all_images[index])
        detection_image = load_image(detection_image_path, max_size=(400, 300))
        canvas2.itemconfig(image_on_canvas2, image=detection_image)


        segmentation_flame_smoke_path = find_matching_image(segmentation_visualized_images_path,
                                                      os.path.splitext(all_images[index])[0])
        segmentation_flame_smoke_image = load_image(segmentation_flame_smoke_path, max_size=(400, 300))
        canvas3.itemconfig(image_on_canvas3, image= segmentation_flame_smoke_image)
        # Update segmentation image
        # segmentation_smoke_path = find_matching_image(segmentation_visualized_path2,
        #                                               os.path.splitext(all_images[index])[0])
        # segmentation_smoke = load_image(segmentation_smoke_path, max_size=(400, 300))

        flame_orientation_path = find_matching_image(flame_orientation_images_path, os.path.splitext(all_images[index])[0])
        if flame_orientation_path is not None and os.path.exists(flame_orientation_path):
            flame_orientation_image = load_image(flame_orientation_path, max_size=(400, 300))
        else:
            # If no segmentation image is found, create a black image of the same size
            flame_orientation_image = create_black_image(400, 300)
            # segmentation_smoke_path = find_matching_image(segmentation_visualized_path2, os.path.splitext(all_images[index])[0])

        canvas4.itemconfig(image_on_canvas4, image=flame_orientation_image)
        # Update references to prevent garbage collection
        results_window.images=[original_image, detection_image,segmentation_flame_smoke_image,flame_orientation_image]


    # Navigation functions
    # Looks like the subwindow data is correct, while the 1-stage data is wrong
    def next_image():
        nonlocal current_index
        current_index = (current_index + 1) % len(all_images)
        update_images(current_index)

    def previous_image():
        nonlocal current_index
        current_index = (current_index - 1) % len(all_images)
        update_images(current_index)


root.mainloop()


