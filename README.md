# Optical flow guided Dynamic 3D Gaussians for novel dynamic scene synthesis

Based on [Dynamic 3D Gaussians](https://dynamic3dgaussians.github.io/). Follow the instructions on the paper's [code base to download the data](https://github.com/JonathonLuiten/Dynamic3DGaussians). 

## Instructions to get segmentation masks from GroundingDINO and SAM
The code to generate segmentation masks is inside of the sam_gaussians branch, therefore you need to switch to that branch in order to generate them.  You will need to download [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and [SAM](https://github.com/facebookresearch/segment-anything) to get started with segmentation mask generation. After downloading these, go inside the subdirectory GroundingDINO and create a folder called weights. You should then proceed by executing "wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" and  "wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" to download the pretrained models. Then simply executing "python generate_masks_with_sam.py" should then generate the required files.

## Instructions for the Optical Flow Experiments

You need to setup the repository in a way that is described in the original Dyamic 3D Gaussian repository, including installing the rasterizer and downloading the data.
