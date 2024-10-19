The D6_KPIMeasurement.py requires 2 necessary weight documents ('sam_b.py' and 'best1.pt') to execute:
For SAM weight 'sam_b.pt': https://kuacae-my.sharepoint.com/:u:/g/personal/100062567_ku_ac_ae/ERYwFmtXh25KuHF3lN6EuU0BuAYEDkyn0jAjfy7YdNYIQQ?e=U3lbxt  

For Flare-smoke Fine-tuned YOLOv8-m weight 'best1.pt': https://kuacae-my.sharepoint.com/:u:/g/personal/100062567_ku_ac_ae/EVXyY7Ew_cVHnm3niFhS6NcBg-_R7wVvFwvsAhHQZart6w?e=uyAZUm


The GUI consists of two main levels, each serving distinct purposes:  

	First-level Main Window: 
   - The interface where the user selects input and output directories and determines the type of analysis to be conducted (image-based or video-based). For the current version, image based analysis refers to flare and smoke detection, segmentation, measurements.
<p align="center">
   <img width="231" alt="图片1" src="https://github.com/user-attachments/assets/56bcc487-3e60-439e-afdb-bc818ed984f6">
</p>



	Second-level Main Window: 
   - Displays the processed results across four consoles. These consoles offer additional interaction through sub-windows when double-clicked, providing more detailed data such as segmentation masks, color analysis, flare-smoke ratio, flare orientation.
<p align="center">
   <img width="233" alt="图片2" src="https://github.com/user-attachments/assets/0cb9dfc6-5acc-43d0-a553-673a5f425861">
</p>


1.1 First-level Main Window
The ‘First-level Main Window’ is the starting point of the GUI, where users set up the analysis by selecting directories and defining the analysis scope. This interface includes:
- Directory Selection:
  - Users input the directory paths for images, videos, and output folders through text fields.
  - The directory selection buttons (such as "Select Image Directory," "Select Video Directory," "Select Output Directory" which are marked in red box in the attached screenshot), these buttons can open file dialogs for easy path selection (marked in green box).

<p align="center">
   <img width="232" alt="图片3" src="https://github.com/user-attachments/assets/f90a2747-3254-483a-b900-e49c1768e2ba">
</p>
 

- Analysis Type Selection (marked in red box):
  - The user can start image-based or video-based analysis by clicking the respective buttons. 
  - The "Start Image Auto-labelling" button initiates image-based object detection and segmentation.
  - The "Split Video to Frames" button extracts frames from video files for analysis.
  - The "Start Analysis" button runs gas flare analysis on videos.
 
<p align="center">
    <img width="233" alt="图片4" src="https://github.com/user-attachments/assets/b312e3a5-22a4-4f40-a531-9484204aa81a">
</p>


- Log Console (marked in green box):
  - The log console displays real-time progress of the tasks, including how many images or frames have been processed and whether the tasks completed successfully. If the user selects ‘split video to frames’ service, the Log Console will display how many frames are split from the raw video and also the collapsed time.

1.2 Second-level Main Window
The ‘Second-level Main Window’ opens once the analysis is completed. It contains four consoles, each displaying specific aspects of the processed data. These consoles provide users with an overview of the analysis, while sub-windows can be accessed for more detailed information.
- Console 1: Original Image - Displays the raw, unprocessed image from the input directory.
- Console 2: Detection Results - Shows bounding boxes around detected objects (flame, smoke) and presents metrics such as flame angle, smoke/flare ratio, flame size, and smoke size.
- Console 3: Segmentation (Flame and Smoke) - Highlights flame and smoke areas with colored masks (red for flame, blue for smoke).
- Console 4: Flame Orientation - Shows a graphical representation of the flame’s angle and direction.

 
<p align="center">
     <img width="241" alt="图片5" src="https://github.com/user-attachments/assets/76d22fec-1fbf-4a08-a8e0-5fa4e6274d6b">
</p>



2. Sub-windows Triggered by Consoles
The sub-windows are detailed information windows that open when the user double-clicks on specific consoles. Each sub-window provides additional insights into the corresponding data.
2.1	Sub-window for Segmentation (Triggered from Console 3)
By double-clicking on the Segmentation Console (Console 3), a sub-window opens to provide a more detailed view of the segmentation process. This includes:

- Flare Mask - The segmentation results showing the areas detected as flames.
- Smoke Mask - The segmentation results showing the areas detected as smoke.
This double-click action leads to a sub-window similar to the following visualization:
 
<p align="center">
    <img width="249" alt="图片6" src="https://github.com/user-attachments/assets/85645d78-972c-406e-856a-f988061ae04a">
</p>

2.2	Sub-window for Detailed Metrics (Triggered from Console 4)

By double-clicking on the Flame Orientation Console (Console 4), another sub-window appears, providing the following detailed metrics (Marked in Red box):

- Flare/Smoke Ratio- A graphical representation of the flare-to-smoke ratio, flare-to-image ratio, and smoke-to-image ratio.
- Color Analysis - A histogram showing the distribution of color intensity across red, green, and blue channels for the image.
- Flame and Smoke Sizes - The calculated sizes of the flame and smoke regions.
- Flame Angle - The angle of the flame, with details on how the flame is oriented in the image.+
An example of this sub-window with detailed metrics is shown below:
 <p align="center">
    <img width="284" alt="图片7" src="https://github.com/user-attachments/assets/4550d102-60bd-41c7-aca4-91ada277847f">
</p>



