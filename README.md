# Pixel-level annotator

This is a simple tool to annotate music sheet images pixel by pixel. It is useful for creating segmentation masks.
 
There are 4 layers:
- **Background**
- **Staff**
- **Notes**
- **Lyrics**

## Usage

Place your images in the `images` folder. Run the script `main.py` and start annotating.

Annotate the pixels by clicking on the image. You can also annotate all neighboring pixels with the same label by right-clicking. The threshold can be adjusted with the slider. If the pixels are already annotated with the selected label, they are relabeled.

Annotations for the selected layer are shown in red, and all other layers are shown in blue (if they are not hidden).

There is a progress bar at the top of the window, the goal is to reach 100% completion.

Annotations are saved in the `annotations` folder. Annotations are saved as grayscale images, one for each layer. E.g. `image.png` will have the following annotations:
- `image_0.png`: background
- `image_1.png`: staff
- `image_2.png`: notes
- `image3_.png`: lyrics

Images are saved after each annotation. You can stop the application at any time and continue later.

## Controls

- **Left click**: Annotate pixel with the selected label
- **Right click**: Annotate all neighboring pixels
- **Space (hold)**: Hide annotations
- **1-4**: Select label
- **i**: Show/hide image
- **o**: Show/hide other layers
- **+**: Zoom in
- **-**: Zoom out
- **Ctrl + z**: Undo
- **Shift + wheel**: Horizontal scroll
