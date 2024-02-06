# Optical flow guided Dynamic 3D Gaussians for novel dynamic scene synthesis

Based on [Dynamic 3D Gaussians](https://dynamic3dgaussians.github.io/) 

Follow the instructions on [this repo](https://github.com/JonathonLuiten/Dynamic3DGaussians) to download the rasterizer, install dependencies, and get the data. 

## Getting started with Segmentation Masks
You will need to install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and [SAM](https://github.com/facebookresearch/segment-anything) to get started with segmentation mask generation. After installing them, go inside the subdirectory GroundingDINO and create a folder called weights. You should then proceed by executing "wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" and  "wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" to download the pretrained models. Then simply executing "python generate_masks_with_sam.py" should then generate the required files.

To train and evaluate model, simply execute "python train.py" and "python execute.py", respectively.
