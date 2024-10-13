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

    def subwindow3(event):
        index = current_index
        masks_window = tk.Toplevel(root)
        masks_window.title("Flame and Smoke Masks")

        frame = ttk.Frame(masks_window)
        frame.grid(row=0, column=0)

        segmentation_flame_path = find_matching_image(segmentation_visualized_path1, os.path.splitext(all_images[index])[0])
        segmentation_smoke_path = find_matching_image(segmentation_visualized_path2, os.path.splitext(all_images[index])[0])

        if segmentation_flame_path and os.path.exists(segmentation_flame_path):
            flame_mask_image = load_image(segmentation_flame_path, max_size=(400, 300))
            label_flame_mask = tk.Label(frame, text="Flame Mask")
            label_flame_mask.grid(row=0, column=0)
            canvas_flame_mask = tk.Canvas(frame, width=400, height=300)
            canvas_flame_mask.create_image(20, 20, anchor=tk.NW, image=flame_mask_image)
            canvas_flame_mask.grid(row=1, column=0)
            canvas_flame_mask.image = flame_mask_image

        if segmentation_smoke_path and os.path.exists(segmentation_smoke_path):
            smoke_mask_image = load_image(segmentation_smoke_path, max_size=(400, 300))
            label_smoke_mask = tk.Label(frame, text="Smoke Mask")
            label_smoke_mask.grid(row=0, column=1)
            canvas_smoke_mask = tk.Canvas(frame, width=400, height=300)
            canvas_smoke_mask.create_image(20, 20, anchor=tk.NW, image=smoke_mask_image)
            canvas_smoke_mask.grid(row=1, column=1)
            canvas_smoke_mask.image = smoke_mask_image

        ########## put the mask

        # if segmentation_smoke_path and os.path.exists(segmentation_smoke_path):
        #     smoke_mask_image = load_image(segmentation_smoke_path, max_size=(400, 300))
        #     label_smoke_mask = tk.Label(frame, text="Smoke Mask")
        #     label_smoke_mask.grid(row=0, column=1)
        #     canvas_smoke_mask = tk.Canvas(frame, width=400, height=300)
        #     canvas_smoke_mask.create_image(20, 20, anchor=tk.NW, image=smoke_mask_image)
        #     canvas_smoke_mask.grid(row=1, column=1)
        #     canvas_smoke_mask.image = smoke_mask_image
        #
        # if segmentation_smoke_path and os.path.exists(segmentation_smoke_path):
        #     smoke_mask_image = load_image(segmentation_smoke_path, max_size=(400, 300))
        #     label_smoke_mask = tk.Label(frame, text="Smoke Mask")
        #     label_smoke_mask.grid(row=0, column=1)
        #     canvas_smoke_mask = tk.Canvas(frame, width=400, height=300)
        #     canvas_smoke_mask.create_image(20, 20, anchor=tk.NW, image=smoke_mask_image)
        #     canvas_smoke_mask.grid(row=1, column=1)
        #     canvas_smoke_mask.image = smoke_mask_image

    def create_colorbar(ratio):
        # Create a figure and a single subplot
        fig, ax = plt.subplots(figsize=(2, 6))
        # Remove axis
        ax.set_axis_off()
        # Create a horizontal color bar
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 1)),
                            ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
        # Set the color bar label
        cbar.set_label('Combustion health')
        # Set the ticks and tick labels
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['0', '0.5', '1'])
        # Highlight the current ratio with an annotation
        ax.annotate(f'Current Ratio: {ratio:.2f}', xy=(ratio, 0.5), xytext=(ratio, 0.6),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        return fig

    def update_colorbar(cbar, ratio):
        # 更新colorbar颜色
        cbar.set_clim(0, 1)
        cbar.set_cmap(plt.cm.RdYlGn)
        cbar.draw_all()

    def subwindow4(event):
        index = current_index
        measure_window = tk.Toplevel(root)
        measure_window.title("Measurement, add RGB")

        frame = ttk.Frame(measure_window)
        frame.grid(row=0, column=0)

        # 找到匹配的文件路径
        flame_size_pth = find_matching_image(value_path1, os.path.splitext(all_images[index])[0])
        smoke_size_pth = find_matching_image(value_path2, os.path.splitext(all_images[index])[0])
        flame_smoke_ratio_pth = find_matching_image(value_path3, os.path.splitext(all_images[index])[0])
        flame_orientation_pth = find_matching_image(value_path4, os.path.splitext(all_images[index])[0])

        # 读取文件中的数据
        value1, value2, value3, value4 = [], [], [], []

        with open(flame_size_pth, 'r') as file:
            value1 = [line.strip() for line in file.readlines()]

        with open(smoke_size_pth, 'r') as file:
            value2 = [line.strip() for line in file.readlines()]

        with open(flame_smoke_ratio_pth, 'r') as file:
            for line in file.readlines():
                value3.extend(line.strip().split(' '))

        with open(flame_orientation_pth, 'r') as file:
            value4 = [line.strip() for line in file.readlines()]

        # 创建直方图
        # 绘制柱状图
        fig, ax_bar = plt.subplots(figsize=(5, 4))
        x_value = ['smoke/flare ratio', 'flare/image ratio', 'smoke/image ratio']
        y_value = [float(value3[2]), float(value3[0]), float(value3[1])]
        ax_bar.bar(x_value, y_value)
        ax_bar.tick_params(axis='both', labelsize=8)
        ax_bar.set_title('flare smoke ratio')
        ax_bar.set_xlabel('name')
        ax_bar.set_ylabel('ratio value')

        canvas_bar = FigureCanvasTkAgg(fig, master=frame)
        canvas_bar.draw()
        canvas_bar.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)

        #########################
        # 读取检测图像
        detection_image_path = os.path.join(detection_images_path, all_images[index])
        rgbimage = cv2.imread(detection_image_path)
        image_rgb = cv2.cvtColor(rgbimage, cv2.COLOR_BGR2RGB)

        # Flatten the image into a single array of RGB tuples
        pixels = image_rgb.reshape((-1, 3))

        # 创建颜色直方图
        fig_color, ax_color = plt.subplots(figsize=(5, 4))
        num_bins = 256
        colors = ['red', 'green', 'blue']
        for i in range(3):  # 循环处理RGB通道
            ax_color.hist(pixels[:, i], bins=num_bins, color=colors[i], alpha=0.7,
                          label=f'{colors[i].capitalize()} Channel')

        # 添加标题和标签
        ax_color.set_title('Color Analysis')
        ax_color.set_xlabel('Intensity')
        ax_color.set_ylabel('Frequency')
        ax_color.legend()

        canvas_color = FigureCanvasTkAgg(fig_color, master=frame)
        canvas_color.draw()
        canvas_color.get_tk_widget().grid(row=1, column=0, padx=10, pady=10)
        ########################
        # 显示文本结果
        text_frame = ttk.Frame(measure_window)
        text_frame.grid(row=2, column=0, padx=10, pady=10)

        label5 = tk.Label(text_frame, text="flame size: " + value1[0] + "px  smoke size: " + value2[0])
        label5.grid(row=0, column=0, padx=5, pady=5)

        label6 = tk.Label(text_frame, text="smoke area ratio: " + value3[2])
        label6.grid(row=1, column=0, padx=5, pady=5)

        label7 = tk.Label(text_frame, text="flame's angle: " + value4[0] + " (Rotate clockwise on the x-negative axis)")
        label7.grid(row=2, column=0, padx=5, pady=5)

        # 防止垃圾回收的引用更新
        measure_window.images = [fig, fig_color]

    first_image_filename = get_first_image(folder_path)
    original_image_path = os.path.join(folder_path, first_image_filename) if first_image_filename else None

    # Construct paths for the corresponding images in the output folders
    detection_image_path = os.path.join(detection_images_path, first_image_filename) if first_image_filename else None

    # Find a matching image in the segmentation_visualized_path
    segmentation_flame_smoke_path = find_matching_image(segmentation_visualized_images_path, os.path.splitext(first_image_filename)[
        0]) if first_image_filename else None
    flame_orientation_path = find_matching_image(flame_orientation_images_path, os.path.splitext(first_image_filename)[
        0]) if first_image_filename else None
    # Create a new Toplevel window
    results_window = tk.Toplevel(root)
    results_window.title("Results Visualizer")
    results_window.geometry("830x700")  # 可以控制窗口数量。2*2，3*2，3*3...都可以

    if original_image_path:
        original_image = load_image(original_image_path, max_size=(400, 300))
        canvas1 = tk.Canvas(results_window, width=400, height=300)
        canvas1.create_image(20, 20, anchor=tk.NW, image=original_image)
        #canvas1.create_text(200, 290, anchor='n', text="Original Image",font=('Arial', 12), fill='white', justify='center')
        canvas1.grid(row=0, column=0, padx=0, pady=0)
        label1 = tk.Label(results_window, text="Original Image")
        label1.grid(row=1, column=0,padx=0,pady=0)

    if os.path.exists(detection_image_path):
        detection_image = load_image(detection_image_path, max_size=(400, 300))
        canvas2 = tk.Canvas(results_window, width=400, height=300)
        canvas2.create_image(20, 20, anchor=tk.NW, image=detection_image)
        #canvas2.create_text(200, 290, anchor='n', text="Detection",font=('Arial', 12), fill='white', justify='center')
        canvas2.grid(row=0, column=1, padx=0, pady=0)
        label2 = tk.Label(results_window, text="Detection")
        #canvas2.create_text(0, 300, anchor='sw', text="Description 1")
        label2.grid(row=1, column=1,padx=0,pady=0)

    if segmentation_flame_smoke_path:
        segmentation_flame_smoke_image = load_image(segmentation_flame_smoke_path, max_size=(400, 300))
        canvas3 = tk.Canvas(results_window, width=400, height=300)
        canvas3.create_image(20, 20, anchor=tk.NW, image=segmentation_flame_smoke_image)
        canvas3.grid(row=2, column=0, padx=0, pady=0)
        canvas3.bind("<Double-1>", subwindow3)  # Bind double-click event to canvas3
        label3 = tk.Label(results_window, text="Segmentation_flame_smoke")
        label3.grid(row=3, column=0, padx=0, pady=0)

    if flame_orientation_path and os.path.exists(flame_orientation_path):
        flame_orientation_image = load_image(flame_orientation_path, max_size=(400, 300))
        canvas4 = tk.Canvas(results_window, width=400, height=300)
        canvas4.create_image(20, 20, anchor=tk.NW, image=flame_orientation_image)
        canvas4.grid(row=2, column=1, padx=0, pady=0)
        canvas4.bind("<Double-1>", subwindow4)  # Bind double-click event to canvas4
        label4 = tk.Label(results_window, text="Flame Orientation")
        label4.grid(row=3, column=1, padx=0, pady=0)
    else:
        flame_orientation_image = create_black_image(400, 300)


    previous_button = ttk.Button(results_window, text="Previous Image", command=previous_image)
    previous_button.grid(row=4, column=0, padx=10, pady=10)

    next_button = ttk.Button(results_window, text="Next Image", command=next_image)
    next_button.grid(row=4, column=1, padx=10, pady=10)

    all_images = get_all_images(folder_path)
    current_index = 0
#################################


    image_on_canvas1 = canvas1.create_image(20, 20, anchor=tk.NW, image=original_image)
    image_on_canvas2 = canvas2.create_image(20, 20, anchor=tk.NW, image=detection_image)
    image_on_canvas3 = canvas3.create_image(20, 20, anchor=tk.NW, image=segmentation_flame_smoke_image)
    image_on_canvas4 = canvas4.create_image(20, 20, anchor=tk.NW, image=flame_orientation_image)
    # image_on_canvas3 = canvas3.create_image(20, 20, anchor=tk.NW, image=segmentation_flame)
    # image_on_canvas4 = canvas4.create_image(20, 20, anchor=tk.NW, image=segmentation_smoke)

    # Get all images in the folder
    all_images = get_all_images(folder_path)
    current_index = 0  # Index of the currently displayed image
    # Create navigation buttons
    next_button = ttk.Button(results_window, text="Next Image", command=next_image)
    next_button.grid(row=4, column=1,padx=15,pady=15)
    #next_button.pack()

    prev_button = ttk.Button(results_window, text="Previous Image", command=previous_image)
    prev_button.grid(row=4, column=0,padx=15,pady=15)
    #prev_button.pack()

    ## ADDED TO HERE ##

    # Keep references to the images to prevent garbage collection
    results_window.images = [original_image, detection_image, segmentation_flame_smoke_image,flame_orientation_image]


def draw_yolo_labels():
    # Create a subfolder for detection images
    image_folder = entry_dir1.get()
    output_folder = entry_dir2.get()

    label_folder = os.path.join(output_folder, "Raw_Detection_labels")
    os.makedirs(label_folder, exist_ok=True)  # Ensure label folder exists

    detection_images_folder = os.path.join(output_folder, "Detection_Images")
    os.makedirs(detection_images_folder, exist_ok=True)

    # Iterate over all image files in the image folder
    for image_file in os.listdir(image_folder):
        # Check for image file extension
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(label_folder, label_file)

            # Check if the corresponding label file exists
            if not os.path.exists(label_path):
                print(f"Label file for {image_file} does not exist. Skipping.")
                continue

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image {image_file}. Skipping.")
                continue

            # Get image dimensions
            img_height, img_width = image.shape[:2]

            # Read YOLO labels from the label file
            with open(label_path, 'r') as file:
                lines = file.readlines()

                for line in lines:
                    # Parse YOLO format (class_id, x_center, y_center, width, height)
                    class_id, x_center, y_center, width, height = map(float, line.split())

                    # Convert normalized positions to pixel values
                    x_center, y_center, width, height = (x_center * img_width, y_center * img_height,
                                                         width * img_width, height * img_height)
                    x_min = int(x_center - (width / 2))
                    y_min = int(y_center - (height / 2))

                    # Draw rectangle
                    if int(class_id) == 0:
                        cv2.rectangle(image, (x_min, y_min), (x_min + int(width), y_min + int(height)), (0, 0, 255), 2)
                        cv2.putText(image, str("Flame"), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(image, (x_min, y_min), (x_min + int(width), y_min + int(height)), (255, 0, 0), 2)
                        cv2.putText(image, str("Smoke"), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, (255, 0, 0), 2)

            # Adjust the output path to save in the Detection_Images subfolder
            output_image_path = os.path.join(detection_images_folder, image_file)
            cv2.imwrite(output_image_path, image)
    print(f"Processed and saved Detection Images and raw labels.")
    print(f"Processed and saved Segmentation Images and raw labels.")


# Usage example:
# draw_yolo_labels('path_to_image_folder', 'path_to_label_folder', 'path_to_output_folder')


def open_new_window():
    # Create a new Toplevel window
    new_window = tk.Toplevel(root)
    new_window.title("About This Tool")
    new_window.geometry("600x160")  # Adjusted the size of the window

    # Multiline text for the label
    text = """This tool was developed under the project titled "Vision-based Flare Analytics"

     in a collaboration between Khalifa University and ADNOC.

The tool was developed by Muaz Al Radi (KU), Pengfei Li (KU),and Mu Xing (KU)

 under the supervision of Prof. Naoufel Werghi."""

    # Add a label with multiline text to the new window
    label = tk.Label(new_window, text=text, justify=tk.CENTER)
    label.pack(padx=10, pady=20)

######################## Add the function of loading video ############################

##########################################################################################


def select_directory1():
    directory1 = filedialog.askdirectory()
    entry_dir1.delete(0, tk.END)
    entry_dir1.insert(0, directory1)


def select_directory2():
    directory2 = filedialog.askdirectory()
    entry_dir2.delete(0, tk.END)
    entry_dir2.insert(0, directory2)

def select_directory3():
    directory3 = filedialog.askdirectory()
    entry_dir3.delete(0, tk.END)
    entry_dir3.insert(0, directory3)


def save_directories():
    dir1 = entry_dir1.get()
    dir2 = entry_dir2.get()
    dir3 = entry_dir3.get()
    # Here you can save these directories to a file or use them further in the code
    print("Directory 1:", dir1)
    print("Directory 2:", dir2)
    print("Directory 3:", dir3)


class PrintLogger:
    def __init__(self, textbox):
        self.textbox = textbox

    def write(self, text):
        self.textbox.configure(state='normal')  # Enable the log box to insert text
        self.textbox.insert(tk.END, text)  # Write text to the textbox
        self.textbox.see(tk.END)  # Scroll to end
        self.textbox.configure(state='disabled')  # Disable the log box to prevent user editing

    def flush(self):
        pass


def run_yolo_detector(image_path, det_model_path, sam_model_path, output_path):
    # 只是合并了，还是全图检测
    # TBD： YOLO框+SAM，以及对应的可视化工具
    # Load the models
    det_model = YOLO(det_model_path)
    sam_model = SAM(sam_model_path)

    # Create subfolders for output
    ####### for generating numerical results
    raw_detection_labels_path = os.path.join(output_path, "Raw_Detection_labels")
    raw_segmentation_labels_path1 = os.path.join(output_path, "Raw_Segmentation_flame_labels")
    raw_segmentation_labels_path2= os.path.join(output_path, "Raw_Segmentation_smoke_labels")
    segmented_flame_path = os.path.join(output_path, "Segmentation_Visualized_flame")
    segmented_smoke_path = os.path.join(output_path, "Segmentation_Visualized_smoke")
    flame_size_path = os.path.join(output_path, "flame_size")
    smoke_size_path = os.path.join(output_path, "smoke_size")
    flame_smoke_ratio_path = os.path.join(output_path, "flame_smoke_ratio")
    flame_orientation_path = os.path.join(output_path, "flame_orientation")  # for measurement

    #### for generating combined flare-smoke visualization and orientation

    # empty_seg_filename1 = os.path.splitext(os.path.basename(image_path))[0] + "_empty.txt"
    segmentation_visualized_images_path = os.path.join(output_path, "Segmentation_Visualized_flame_smoke")
    flame_orientation_images_path = os.path.join(output_path, "Segmentation_Visualized_flame_orientation")

    ########## here we made the directory and store the images

    os.makedirs(raw_detection_labels_path, exist_ok=True)

    os.makedirs(raw_segmentation_labels_path1, exist_ok=True)  # flare label
    os.makedirs(raw_segmentation_labels_path2, exist_ok=True)  # smoke label
    os.makedirs(segmented_flame_path, exist_ok=True)  # flare mask
    os.makedirs(segmented_smoke_path, exist_ok=True)  # smoke mask
    os.makedirs(flame_size_path, exist_ok=True)
    os.makedirs(smoke_size_path, exist_ok=True)
    os.makedirs(flame_smoke_ratio_path, exist_ok=True)
    os.makedirs(flame_orientation_path, exist_ok=True)


    os.makedirs(segmentation_visualized_images_path, exist_ok=True)

    os.makedirs(flame_orientation_images_path, exist_ok=True)


    # Read the input image
    image = Image.open(image_path)
    img_width=image.width
    img_height=image.height

    # Convert PIL image to OpenCV format
    image_cv = np.array(image)
    image = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)  # possible to add RGB analysis here

    # Perform object detection
    results = det_model(image, stream=True)
    empty_seg_filename1="none.txt"

    # Open the output file for YOLO formatted data
    yolo_labels_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    value_filename=os.path.splitext(os.path.basename(image_path))[0]
    with open(os.path.join(raw_detection_labels_path, yolo_labels_filename), "w") as output_file:
        masks_found = False
        for result in results:
            # Initialize flags to check if a class has been detected
            flame_size = 0
            smoke_size = 0
            smoke_area_ratio = 0
            flame_image_ratio = 0
            smoke_image_ratio = 0
            flame_orientation = 0
            box_max = 0
            temp1 = 0
            temp2 = 0
            boxes = result.boxes.xyxy  # Bounding boxes
            class_ids = result.boxes.cls.int().tolist()  # Class IDs

            # Check if there are any detections
            if boxes.numel() == 0:
                # No masks found, save original image
                img_height, img_width = image.size
                empty_img = np.zeros((img_width, img_height, 3), dtype=np.uint8)
                empty_img_filename = os.path.splitext(os.path.basename(image_path))[0] + "_empty.png"
                empty_flare_path1 = os.path.join(segmented_flame_path, empty_img_filename)  # change
                empty_smoke_path2 = os.path.join(segmented_smoke_path, empty_img_filename)  # change
                cv2.imwrite(empty_flare_path1, empty_img)
                cv2.imwrite(empty_smoke_path2, empty_img)

                # Also save an empty .txt file for segmentation labels
                empty_seg_filename = os.path.splitext(os.path.basename(image_path))[0] + "_segmentation_empty.txt"
                empty_seg_path1 = os.path.join(raw_segmentation_labels_path1, empty_seg_filename)
                empty_seg_path2 = os.path.join(raw_segmentation_labels_path2, empty_seg_filename)


                empty_img_path1 = os.path.join(segmentation_visualized_images_path, empty_img_filename)
                empty_img_path2 = os.path.join(flame_orientation_images_path, empty_img_filename)
                cv2.imwrite(empty_img_path1, image)
                cv2.imwrite(empty_img_path2, image)

                ################
                empty_seg_filename1 = os.path.splitext(os.path.basename(image_path))[0] + "_empty.txt"
                empty_seg_path3 = os.path.join(flame_size_path, empty_seg_filename1)
                empty_seg_path4 = os.path.join(smoke_size_path, empty_seg_filename1)
                empty_seg_path5 = os.path.join(flame_smoke_ratio_path, empty_seg_filename1)
                empty_seg_path6 = os.path.join(flame_orientation_path, empty_seg_filename1)
                open(empty_seg_path1, 'w').close()
                open(empty_seg_path2, 'w').close()
                with open(empty_seg_path3, "w") as f1:
                    f1.write(f" {flame_size:.2f} \n")  # 怀疑这个思泽是看YOLO框大小
                with open(empty_seg_path4, "w") as f2:
                    f2.write(f" {smoke_size:.2f} \n")
                with open(empty_seg_path5, "w") as f3:
                    f3.write(f" {flame_image_ratio:.2f} {smoke_image_ratio:.2f} {smoke_area_ratio:.2f} \n")
                with open(empty_seg_path6, "w") as f4:
                    f4.write(f" {flame_orientation:.2f} \n")

                # Also save an empty .txt file for segmentation labels
                empty_txt_filename = os.path.splitext(os.path.basename(image_path))[0] + "_empty.txt"
                empty_detect_path=os.path.join( raw_detection_labels_path, empty_txt_filename)
                open(empty_detect_path,'w').close()

                continue  # No detections, continue with the next result
            # Iterate through each detected object
            area = 0
            area_flame = []
            area_smoke = []
            temp1=0
            temp2=0
            for i, box in enumerate(boxes):
                x_min, y_min, x_max, y_max = map(int, box.tolist())
                class_id = class_ids[i]
                img_height,img_width,_ = image.shape
                x_center = ((x_min + x_max) / 2) / img_width
                y_center = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                # Write the detection in YOLO format to the output file
                output_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                if class_id == 0:

                    flame_size = (x_max - x_min)*(y_max - y_min)+flame_size
                    # flare size 目前是bounding box 大小， 而不是分割图像的像素数量
                    box_size=(x_max - x_min)*(y_max - y_min)
                    # calculate slope
                    if box_size>box_max:
                        if x_min - x_max != 0:
                            slope = (y_max - y_min) / (x_max - x_min)
                        else:
                            slope = float('inf')  # other situation

                            # calculate angle
                        angle_rad = math.atan(slope)
                        # 这里的角度是计算bound ing box 的斜率

                        # convert
                        flame_orientation = math.degrees(angle_rad)

                    ###############之后没对照了##################
                    ###############之后没对照了##################

                    temp1=temp1+1
                    x_min, y_min, x_max, y_max = map(int, box.tolist())
                    w = (x_max - x_min)
                    h = (y_max - y_min)
                    center = (w // 2, h // 2)
                    area1 = w * h
                    if area1 > area:
                        area = area1
                        x_min1, y_min1, x_max1, y_max1 = x_min, y_min, x_max, y_max
                        w1, h1 = w, h

                    ################## I add Muaz before xing ##########
                    ################################################################################################################################################
                    ################################################################################################################################################
                    #####################################################The end of Muaz numerical results################################################################
                #
                    # segmented_result = sam_model(image, points=[(x_min + x_max) / 2, (y_min + y_max) / 2],
                    #                              labels=[class_id])

                    segmentation_result = sam_model(image, points=[(x_min + x_max) / 2, (y_min + y_max) / 2],
                                                 labels=[class_id])

                    # Check if masks are available in the result
                    if segmentation_result[0].masks is not None:
                        masks = segmentation_result[0].masks.data.cpu().numpy()
                        masks_found = True  # Indicate that at least one mask was found

                        # Process each mask
                        for i, mask in enumerate(masks):
                            # Ensure mask is 2D and convert to uint8
                            mask_2d = mask.squeeze()  # Remove any extra dimensions
                            mask_cv = np.uint8(mask_2d * 255)

                            # Find contours of the mask，
                            # contour function here is to draw the mask. Again modify it.
                            contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            # Construct filename for raw segmentation labels
                            raw_segmentation_filename = os.path.splitext(os.path.basename(image_path))[
                                                            0] + f"_segmentation_{i}.txt"
                            raw_segmentation_path = os.path.join(raw_segmentation_labels_path1,
                                                                 raw_segmentation_filename)

                            # Open the segmentation label file
                            with open(raw_segmentation_path, "w") as seg_file:
                                seg_file.write(f"{class_id}")
                                for cnt in contours:
                                    for point in cnt:
                                        x, y = point[0]
                                        # Normalize the coordinates
                                        normalized_x = x / img_width
                                        normalized_y = y / img_height
                                        seg_file.write(f" {normalized_x:.6f} {normalized_y:.6f}")
                                        # seg_file.write(f" {x} {y}")
                                seg_file.write("\n")

                            # Apply the mask to the image
                            segmented_img = cv2.bitwise_and(image_cv, image_cv, mask=mask_cv)

                            # Construct the filename for the segmented image
                            segmented_filename = os.path.splitext(os.path.basename(image_path))[
                                                     0] + f"_segmented_{i}.png"
                            segmented_flame_path = os.path.join(segmented_flame_path, segmented_filename)

                            # Save the segmented image
                            cv2.imwrite(segmented_flame_path, segmented_img)
                if class_id == 1:
                    smoke_size = (x_max - x_min) * (y_max - y_min) + smoke_size
                    temp2 = temp2 + 1
                    # Apply SAM to the image
                    segmented_result = sam_model(image, points=[(x_min + x_max) / 2, (y_min + y_max) / 2],
                                                 labels=[class_id])

                    # Check if masks are available in the result
                    if segmented_result[0].masks is not None:
                        masks = segmented_result[0].masks.data.cpu().numpy()
                        # segment results,直接根据SAM 模型反向画出
                        masks_found = True  # Indicate that at least one mask was found

                        # Process each mask
                        for i, mask in enumerate(masks):
                            # Ensure mask is 2D and convert to uint8
                            mask_2d = mask.squeeze()  # Remove any extra dimensions
                            mask_cv = np.uint8(mask_2d * 255)

                            # Find contours of the mask
                            contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            # Construct filename for raw segmentation labels
                            raw_segmentation_filename = os.path.splitext(os.path.basename(image_path))[
                                                            0] + f"_segmentation_{i}.txt"
                            raw_segmentation_path = os.path.join(raw_segmentation_labels_path2,
                                                                 raw_segmentation_filename)

                            # Open the segmentation label file
                            with open(raw_segmentation_path, "w") as seg_file:
                                seg_file.write(f"{class_id}")
                                for cnt in contours:
                                    for point in cnt:
                                        x, y = point[0]
                                        # Normalize the coordinates
                                        normalized_x = x / img_width
                                        normalized_y = y / img_height
                                        seg_file.write(f" {normalized_x:.6f} {normalized_y:.6f}")
                                        # seg_file.write(f" {x} {y}")
                                seg_file.write("\n")

                            # Apply the mask to the image
                            segmented_img = cv2.bitwise_and(image_cv, image_cv, mask=mask_cv)

                            # Construct the filename for the segmented image
                            segmented_filename = os.path.splitext(os.path.basename(image_path))[
                                                     0] + f"_segmented_{i}.png"
                            segmented_smoke_path = os.path.join(segmented_smoke_path, segmented_filename)

                            # Save the segmented image
                            cv2.imwrite(segmented_smoke_path, segmented_img)

                flame_image_ratio = flame_size / (img_width * img_height)
                smoke_image_ratio = smoke_size / (img_width * img_height)
                smoke_area_ratio = smoke_size / (flame_size + smoke_size)
                # flame_smoke_ratio = flame_size / (smoke_size + 1e-6)
                # define empty cases

                # if temp2 == 0:
                #     img_height, img_width = image.size
                #     empty_img = np.zeros((img_width, img_height, 3), dtype=np.uint8)
                #     empty_img_filename = os.path.splitext(os.path.basename(image_path))[0] + "_empty.png"
                #     empty_img_path2 = os.path.join(segmented_smoke_path, empty_img_filename)
                #     cv2.imwrite(empty_img_path2, empty_img)
                #     # Also save an empty .txt file for segmentation labels
                #     empty_seg_filename = os.path.splitext(os.path.basename(image_path))[0] + "_segmentation_empty.txt"
                #     empty_seg_path2 = os.path.join(raw_segmentation_labels_path2, empty_seg_filename)
                #     open(empty_seg_path2, 'w').close()
                #     # flame_smoke_ratio=1
                #     flame_image_ratio = 1
                #     smoke_image_ratio = 1
                #     smoke_area_ratio = 1
                #
                # if not (temp1 or temp2):
                #     flame_image_ratio = 0
                #     smoke_image_ratio = 0
                #     smoke_area_ratio = 0
                #
                #     # flame_smoke_ratio = 0

                if temp1 or temp2:
                    # if yolo_labels_filename.lower() not in empty_seg_filename1.lower():
                    with open(os.path.join(flame_size_path, yolo_labels_filename), "w") as output_file1:
                        output_file1.write(f" {flame_size:.2f} \n")
                    with open(os.path.join(smoke_size_path, yolo_labels_filename), "w") as output_file2:
                        output_file2.write(f" {smoke_size:.2f} \n")
                    with open(os.path.join(flame_smoke_ratio_path, yolo_labels_filename), "w") as output_file3:
                        output_file3.write(
                            f" {flame_image_ratio:.2f} {smoke_image_ratio:.2f} {smoke_area_ratio:.2f} \n")
                    with open(os.path.join(flame_orientation_path, yolo_labels_filename), "w") as output_file4:
                        output_file4.write(f" {flame_orientation:.2f} \n")

                    ################################################################################################################################################
                    ################################################################################################################################################
                    #####################################################The end of Muaz numerical results################################################################

                #     # segmentation_result = sam_model(image, points=[(x_min + x_max) / 2, (y_min + y_max) / 2],
                #     #                                 labels=[0])
                #     segmentation_result = sam_model(image, points=[(x_min + x_max) / 2, (y_min + y_max) / 2],
                #                                     labels=[0])
                    masks = segmentation_result[0].masks.data.cpu().numpy()
                     # the current mask generation is a disaster, need to be the flare orientation one
                    mask_2d = masks.squeeze()  # Remove any extra dimensions
                    mask_cv = np.uint8(mask_2d * 255)
                    # 假设你有一个二值图像的掩码数组 mask_cv
                    area = np.count_nonzero(mask_cv == 255)
                    area_flame.append(area)
                    for row in range(mask_cv.shape[0]):
                        for col in range(mask_cv.shape[1]):
                            if mask_cv[row][col] == 255:
                                image[row][col] = [0, 0, 255]
                if class_id == 1:
                    temp2=temp2+1
                    x_min, y_min, x_max, y_max = map(int, box.tolist())
                    w = (x_max - x_min)
                    h = (y_max - y_min)
                    segmentation_result = sam_model(image, points=[(x_min + x_max) / 2, (y_min + y_max) / 2],
                                                    labels=[1])
                    masks = segmentation_result[0].masks.data.cpu().numpy()
                    mask_2d = masks.squeeze()  # Remove any extra dimensions
                    mask_cv = np.uint8(mask_2d * 255)
                    # 假设你有一个二值图像的掩码数组 mask_cv
                    area = np.count_nonzero(mask_cv == 255)
                    area_smoke.append(area)
                    for row in range(mask_cv.shape[0]):
                        for col in range(mask_cv.shape[1]):
                            if mask_cv[row][col] == 255:
                                 image[row][col] = [255, 0, 0]

                          # Construct filename for raw segmentation labels
            sum_flame = sum(area_flame)
            sum_smoke = sum(area_smoke)
            smoke_flame_ratio = sum_smoke / sum_flame
            # cv2.putText(image, f"Smoke flame ratio: {smoke_flame_ratio:.2f} ", (40,40),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.5,(255,255,255), 2, cv2.LINE_AA)

            segmented_filename = os.path.splitext(os.path.basename(image_path))[
                                                     0] + f"_segmented.png"
            segmented_flame_smoke_path = os.path.join(segmentation_visualized_images_path, segmented_filename)

            # Save the segmented image
            cv2.imwrite(segmented_flame_smoke_path,image)
        if temp1!=0:
            image = cv2.imread(image_path)
            segmentation_result = sam_model(image, points=[(x_min1 + x_max1) / 2, (y_min1 + y_max1) / 2], labels=[0])
            masks = segmentation_result[0].masks.data.cpu().numpy()
            mask_2d = masks.squeeze()  # Remove any extra dimensions
            mask_cv = np.uint8(mask_2d * 255)
            # 读取图像
            # 寻找轮廓
            contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_area = 0
            max_contour = None
            for contour in contours:
                # 计算当前轮廓的面积
                area = cv2.contourArea(contour)
                # 检查当前轮廓的面积是否大于最大面积， 现在就是算轮廓面积而不是检测框面积--> 更加准确
                if area > max_area:
                    max_area = area
                    max_contour = contour

            # for contour in contours:， 贡献就是发现了椭圆是一个可以涵盖大部分轮廓的形状。然后用函数找中心，轴距，角度
            ellipse = cv2.fitEllipse(max_contour)
            # 提取椭圆参数
            center, axes, angle = ellipse
            # print("椭圆角度（度）：", angle)
            major_axis = max(axes)
            minor_axis = min(axes)
            if angle > 90:
                angle = angle - 90
                #(h, w) = image.shape[:2]
                #center = (w // 2, h // 2)
                angle_rad = angle * np.pi / 180.0
                # 计算箭头的起始中止点。以椭圆的长轴两端为标准
                start_point = (
                    int(center[0] + major_axis * np.cos(angle_rad) / 2),
                    int(center[1] + major_axis * np.sin(angle_rad) / 2))
                end_point = (
                    int(center[0] - major_axis * np.cos(angle_rad) / 2),
                    int(center[1] - major_axis * np.sin(angle_rad) / 2))
            elif angle < 90:
                angle = 90 - angle
                #(h, w) = image.shape[:2]
               # center = (w // 2, h // 2)
                angle_rad = angle * np.pi / 180.0
                start_point = (
                    int(center[0] - major_axis * np.cos(angle_rad) / 2),
                    int(center[1] + major_axis * np.sin(angle_rad) / 2))
                end_point = (
                    int(center[0] + major_axis * np.cos(angle_rad) / 2),
                    int(center[1] - major_axis * np.sin(angle_rad) / 2))

            # 绘制箭头线段
            cv2.ellipse(image, ellipse, (0, 255, 0), 2)
            cv2.arrowedLine(image, start_point, end_point, (0, 0, 255), 2)

            ###########################
            def find_matching_image(folder, partial_filename):
                if not partial_filename:
                    return None
                for file in os.listdir(folder):
                    if partial_filename.lower() in file.lower() and file.lower().endswith(
                            ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.txt')):
                        return os.path.join(folder, file)
                return None

            # Function to get a list of all image files in a folder
            def get_all_images(folder):
                return [file for file in os.listdir(folder) if
                        file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            folder_path = entry_dir1.get()
            output_folder = entry_dir2.get()
            all_images = get_all_images(folder_path)
            value_path1 = os.path.join(output_folder, "flame_size")
            value_path2 = os.path.join(output_folder, "smoke_size")
            value_path3 = os.path.join(output_folder, "flame_smoke_ratio")
            value_path4 = os.path.join(output_folder, "flame_orientation")
            index = i
            flame_size_pth = find_matching_image(value_path1, os.path.splitext(all_images[index])[0])
            smoke_size_pth = find_matching_image(value_path2, os.path.splitext(all_images[index])[0])
            flame_smoke_ratio_pth = find_matching_image(value_path3, os.path.splitext(all_images[index])[0])
            flame_orientation_pth = find_matching_image(value_path4, os.path.splitext(all_images[index])[0])

            # 读取文件中的数据
            value1, value2, value3, value4 = [], [], [], []

            with open(flame_size_pth, 'r') as file:
                value1 = [line.strip() for line in file.readlines()]

            with open(smoke_size_pth, 'r') as file:
                value2 = [line.strip() for line in file.readlines()]

            with open(flame_smoke_ratio_pth, 'r') as file:
                for line in file.readlines():
                    value3.extend(line.strip().split(' '))

            with open(flame_orientation_pth, 'r') as file:
                value4 = [line.strip() for line in file.readlines()]
            ############################

            y_offset = 60
            cv2.putText(image, f"Flame angle: {angle:.2f} deg", (1000,y_offset),  # 这里调整字体大小，大一点。
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
            y_offset += 65  # 增加y偏移量，为下一行文本留出空间
            cv2.putText(image, f"Smoke/flare ratio: {value3[2]}", (1000, y_offset),  # 这里调整字体大小，大一点。
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
            y_offset += 65  # 增加y偏移量，为下一行文本留出空间
            cv2.putText(image, f"Flare size: {value1[0]} px", (1000, y_offset),  # 这里调整字体大小，大一点。
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
            y_offset += 65  # 增加y偏移量，为下一行文本留出空间
            cv2.putText(image, f"Smoke size: {value2[0]} px", (1000, y_offset),  # 这里调整字体大小，大一点。
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
            orientation_filename = os.path.splitext(os.path.basename(image_path))[
                                       0] + f"_orientation.png"
            flame_orientation_path = os.path.join(flame_orientation_images_path, orientation_filename)

            # Save the segmented image
            cv2.imwrite(flame_orientation_path, image)
        else:
            image = cv2.imread(image_path)
            orientation_filename = os.path.splitext(os.path.basename(image_path))[
                                       0] + f"_orientation.png"
            flame_orientation_path = os.path.join(flame_orientation_images_path, orientation_filename)

            # Save the segmented image
            cv2.imwrite(flame_orientation_path, image)


# ################### 之后有对照了######################
#
#
# def run_yolo_detector(image_path, det_model_path, sam_model_path, output_path):
#     # Load the models (YOLO and SAM)
#     det_model = YOLO(det_model_path)
#     sam_model = SAM(sam_model_path)
#
#     # Create subfolders for output
#     raw_detection_labels_path = os.path.join(output_path, "Raw_Detection_labels")
#     segmented_flame_path = os.path.join(output_path, "Segmentation_Visualized_flame")
#     segmented_smoke_path = os.path.join(output_path, "Segmentation_Visualized_smoke")
#     flame_orientation_path = os.path.join(output_path, "flame_orientation")  # For measurement
#     segmentation_visualized_images_path = os.path.join(output_path, "Segmentation_Visualized_flame_smoke")
#
#     os.makedirs(raw_detection_labels_path, exist_ok=True)
#     os.makedirs(segmented_flame_path, exist_ok=True)  # Flare mask
#     os.makedirs(segmented_smoke_path, exist_ok=True)  # Smoke mask
#     os.makedirs(segmentation_visualized_images_path, exist_ok=True)
#     os.makedirs(flame_orientation_path, exist_ok=True)
#
#     # Read the input image
#     image = Image.open(image_path)
#     img_width, img_height = image.width, image.height
#     image_cv = np.array(image)
#     image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format
#
#     # Perform object detection (YOLO)
#     results = det_model(image_cv, stream=True)
#
#     # Open the output file for YOLO formatted data
#     yolo_labels_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
#     with open(os.path.join(raw_detection_labels_path, yolo_labels_filename), "w") as output_file:
#         flame_size, smoke_size = 0, 0
#         flame_orientation = 0
#         boxes = results[0].boxes.xyxy  # Bounding boxes
#         class_ids = results[0].boxes.cls.int().tolist()  # Class IDs
#
#         max_flame_area = 0  # To find the largest flame for orientation calculation
#         flame_orientation_contour = None  # To store the contour for the largest flame
#
#         for i, box in enumerate(boxes):
#             x_min, y_min, x_max, y_max = map(int, box.tolist())
#             class_id = class_ids[i]
#             width = x_max - x_min
#             height = y_max - y_min
#
#             # Calculate YOLO labels
#             x_center = ((x_min + x_max) / 2) / img_width
#             y_center = ((y_min + y_max) / 2) / img_height
#             width_norm = width / img_width
#             height_norm = height / img_height
#
#             # Write YOLO detection results
#             output_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
#
#             # Segmentation using SAM for each detected object
#             segmentation_result = sam_model(image_cv, points=[(x_min + x_max) / 2, (y_min + y_max) / 2],
#                                             labels=[class_id])
#
#             if class_id == 0:  # Flame detection
#                 flame_size += width * height
#
#                 # Find the largest flame area for orientation calculation
#                 if width * height > max_flame_area:
#                     max_flame_area = width * height
#                     flame_orientation_contour = segmentation_result[0].masks.data.cpu().numpy().squeeze()
#
#             elif class_id == 1:  # Smoke detection
#                 smoke_size += width * height
#
#             # Save segmentation results (Flame/Smoke)
#             mask_cv = np.uint8(segmentation_result[0].masks.data.cpu().numpy().squeeze() * 255)
#             segmented_img = cv2.bitwise_and(image_cv, image_cv, mask=mask_cv)
#             segmented_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_segmented_{class_id}.png"
#             segmented_path = os.path.join(segmented_flame_path if class_id == 0 else segmented_smoke_path,
#                                           segmented_filename)
#             cv2.imwrite(segmented_path, segmented_img)
#
#         # Calculate smoke/flare ratio
#         smoke_flare_ratio = smoke_size / (flame_size + 1e-6)  # Avoid division by zero
#
#         # Flame orientation calculation (from largest flame)
#         if flame_orientation_contour is not None:
#             # Find contours for the flame
#             contours, _ = cv2.findContours(np.uint8(flame_orientation_contour * 255), cv2.RETR_EXTERNAL,
#                                            cv2.CHAIN_APPROX_SIMPLE)
#             max_contour = max(contours, key=cv2.contourArea)
#             ellipse = cv2.fitEllipse(max_contour)
#             flame_orientation = ellipse[-1]  # Angle in degrees
#
#             # Draw orientation on the original image
#             cv2.ellipse(image_cv, ellipse, (0, 255, 0), 2)
#
#             # Save the orientation result
#             orientation_filename = os.path.splitext(os.path.basename(image_path))[0] + "_orientation.png"
#             cv2.imwrite(os.path.join(flame_orientation_path, orientation_filename), image_cv)
#
#     # Save the combined visualization (smoke and flame)
#     combined_filename = os.path.splitext(os.path.basename(image_path))[0] + "_combined.png"
#     cv2.imwrite(os.path.join(segmentation_visualized_images_path, combined_filename), image_cv)
#
#     # Save flare and smoke size, ratio, and orientation information
#     info_filename = os.path.splitext(os.path.basename(image_path))[0] + "_info.txt"
#     with open(os.path.join(output_path, info_filename), "w") as info_file:
#         info_file.write(f"Flame size: {flame_size}\n")
#         info_file.write(f"Smoke size: {smoke_size}\n")
#         info_file.write(f"Smoke/Flare ratio: {smoke_flare_ratio}\n")
#         info_file.write(f"Flame orientation: {flame_orientation} degrees\n")




def iterate_images_in_folder():  # 调用主函数，predict
    det_model_path = 'best1.pt'  # det_model_path = 'E:\\GUI\\GUI\\bestYolov8x.pt'
    sam_model_path = 'sam_b.pt'

    # Iterate over all files in the folder
    folder_path = entry_dir1.get()
    output_folder = entry_dir2.get()

    # Count processed images
    counter = 0

    for filename in os.listdir(folder_path):
        # Check if the file is an image (you can add more file extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            counter += 1
            image_path = os.path.join(folder_path, filename)

            # Call the run_yolo_detector function
            run_yolo_detector(image_path, det_model_path, sam_model_path, output_folder)
            print(f"Processed {filename}")

    print("----------------------------------------")
    print(f"Processed a total of {counter} images.")


# Function to load an image and return a PhotoImage object
def load_image(path, max_size=(150, 150)):
    image = Image.open(path)
    image.thumbnail(max_size, Image.ANTIALIAS)
    return ImageTk.PhotoImage(image)


##############################
def split_video_frames():
    folder_path = entry_dir3.get()  # dir1 is image input, dir3 is video input, dir2 is output
    output_folder = entry_dir2.get()
    # List video files in the selected directory
    # 初始化计数器和计时器,视频数量，每个视频时长
    total_videos = 0
    total_frames = 0
    total_duration = 0
    start_time = time.time()

    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]

    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_in_video = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total_frames_in_video / fps
        total_duration += duration
        # 统计视频数量
        total_videos += 1

        success, frame = cap.read()
        frame_count = 0
        while success:
            # Process the frame (e.g., save it as an image file)
            frame_path = os.path.join(output_folder, f"{video_file}_frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            # 统计帧数
            total_frames += 1
            # Read the next frame
            success, frame = cap.read()
            frame_count += 1
        # Release the video capture object
        cap.release()
    # 计算耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    # 在 scrolledtext 中显示结果
    log_box.config(state='normal')
    log_box.insert(tk.END, f" \n")
    log_box.insert(tk.END, f" You are using the 'Video Split' service of this GUI.\n")
    log_box.insert(tk.END, f" ####################################################################\n")
    log_box.insert(tk.END, f" There are {total_videos} videos in the directory\n")
    log_box.insert(tk.END, f" The total duration of these videos are: {total_duration:.2f} s\n")
    log_box.insert(tk.END, f" There are {total_frames} frames being split\n")
    log_box.insert(tk.END, f" Total time consumed: {elapsed_time:.2f} s\n")
    log_box.config(state='disabled')


######### Below are for image-level analysis and auto-segmentation
def run_functions():
    iterate_images_in_folder()  # read image roots， and call the main detection functions
    draw_yolo_labels()  # visualize the bounding box on image
    open_results_visualizer_window()  # create consoles
   # result_show_window(), 少了segmentation labels

def resize_image(image_path, new_height):
    image = Image.open(image_path)
    width, height = image.size
    new_width = int((new_height / height) * width)+5
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    return ImageTk.PhotoImage(resized_image)


# Create the main window
root = tk.Tk()
root.title("Autolabelling Tool GUI")

root.tk.call("source", "azure.tcl")
root.tk.call("set_theme", "dark")

kustar_logo = load_image("KUSTAR_Logo.jpg")
kustar_label = tk.Label(root, image=kustar_logo)
kustar_label.grid(row=0, column=0, padx=2, pady=2)

adnoc_logo_resized = resize_image("tn_adnoc.png", 85)
adnoc_label = tk.Label(root, image=adnoc_logo_resized)
adnoc_label.grid(row=0, column=2, padx=0, pady=0)

entry_dir1 = ttk.Entry(root, width=50)
entry_dir1.grid(row=1, column=1, padx=10, pady=10)

entry_dir2 = ttk.Entry(root, width=50)
entry_dir2.grid(row=3, column=1, padx=10, pady=10)

entry_dir3 = ttk.Entry(root, width=50)
entry_dir3.grid(row=2, column=1, padx=10, pady=10)

button_dir1 = ttk.Button(root, text="Select Image Directory", command=select_directory1)
button_dir1.grid(row=1, column=0, padx=10, pady=10)

button_dir2 = ttk.Button(root, text="Select Video Directory", command=select_directory3)
button_dir2.grid(row=2, column=0, padx=10, pady=10)

button_dir3 = ttk.Button(root, text="Select Output Directory", command=select_directory2)
button_dir3.grid(row=3, column=0, padx=10, pady=10)

button_save = ttk.Button(root, text="Start Image Autolabelling", command=run_functions)
button_save.grid(row=4, column=0, padx=3, pady=3)

button_new_window = ttk.Button(root, text="About This Tool", command=open_new_window)
button_new_window.grid(row=1, column=2, padx=3, pady=3)

split_video_button = ttk.Button(root, text="Split Video to Frames", command=split_video_frames)
split_video_button.grid(row=4, column=1, padx=3, pady=3)

video_analysis = ttk.Button(root, text="Start Analysis", command=video_analysis)
video_analysis.grid(row=4, column=2, padx=3, pady=3)

log_box = scrolledtext.ScrolledText(root, state='disabled', height=10)
log_box.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

pl = PrintLogger(log_box)
sys.stdout = pl

root.mainloop()


