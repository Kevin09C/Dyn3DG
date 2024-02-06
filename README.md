# Dynamic 3D Gaussians with Optical Flow Regularization and Segmentation Masks from Natural Langauge Prompts

Based on [Dynamic 3D Gaussians](https://dynamic3dgaussians.github.io/) 

Follow the instructions on [this repo](https://github.com/JonathonLuiten/Dynamic3DGaussians) to download the rasterizer, install dependencies, and get the data. 

##  Segmentation Masks with GroundingDINO and SAM

NOTE: the code to generate the segmentation masks from GroudingDINO and SAM is located on the sam_gaussians branch. We recommend to use that branch to generate segmentation masks.

You will need to install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and [SAM](https://github.com/facebookresearch/segment-anything) to get started with segmentation mask generation. After installing them, go inside the subdirectory GroundingDINO and create a folder called weights. 

You should proceed by downloading the pretrained models, then generating the saegmenation masks.

```bash
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" to download the pretrained models.
python generate_masks_with_sam.py
```



## Segmentation masks from optical flow
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
python scripts/generate_flow.py --dataset_path ${SCENE_DIR} --model weights/raft-sintel.pth
```

Generate masks using optical flow

```bash
python scripts/generate_mask.py --dataset_path ${SCENE_DIR}
```

# Training and evaluating a Dynamic 3D Gaussian model

To train the model execute
```bash
python train.py
```

To train the model including optical flow execute

Check the `train` and `get_loss` function for configuration options and hyperparameters
```bash
python train_optical_flow.py
```

To evaluate a trained model execute 
```bash
python execute.py
```
