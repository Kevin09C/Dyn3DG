# Optical flow guided Dynamic 3D Gaussians for novel dynamic scene synthesis

Based on [Dynamic 3D Gaussians](https://dynamic3dgaussians.github.io/) 

Follow the instructions on [this repo](https://github.com/JonathonLuiten/Dynamic3DGaussians) to download the rasterizer, install dependencies, and get the data. 

## Getting started with Segmentation Masks
You will need to install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and [SAM](https://github.com/facebookresearch/segment-anything) to get started with segmentation mask generation. After installing them, go inside the subdirectory GroundingDINO and create a folder called weights. 

You should then proceed by executing to download the pretrained models and generate the saegmenation masks.

```bash
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" to download the pretrained models.
python generate_masks_with_sam.py
```


To train the model execute "
```bash
python train.py
```
To evaluate a trained model execute 

```bash
python execute.py
```
To generate segmentation masks from the optical flow, follow the following steps. 

First, make sure that you are at the root of the repo, then execute the following commands to download the pretrained optical flow models.

```bash
mkdir weights
cd weights
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip
cd ..
```

Predict optical flows

```bash
python scripts/generate_flow.py --dataset_path ${SCENE_DIR} --model weights/raft-things.pth
```

Generate masks using optical flow

```bash
python scripts/generate_mask.py --dataset_path ${SCENE_DIR}
```
