# -*- coding: utf-8 -*-
"""Dyn3d Sandesh real.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QYwFVixLR4AYablSyAEHgQ9B4B9Tel6Y
"""

import os
dataset_path = "/home/cuedari/Dyn3DG"
sandesh_path = "/home/sharma/Dyn3DG/data"
# In order to access the files in this notebook we have to navigate to the correct folder
os.chdir(dataset_path)
# Check manually if all files are present
# print(sorted(os.listdir()))

# print(os.listdir("data/basketball/flow/1"))
# generate optical flow using RAFT
#!python3 scripts/generate_flow.py --dataset_path data --model weights/models/raft-sintel.pth

# generate masks using Mask RCNN
#!python3 scripts/generate_mask.py --dataset_path data

import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large, raft_small
from torchvision.utils import flow_to_image
import numpy as np
import oflibpytorch as of
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
writer = None 

device = "cuda" if torch.cuda.is_available() else "cpu"
# of_model = raft_small(pretrained=True, progress=False).to(device) # the
# of_model = of_model.eval()

# frame_1 = torchvision.io.read_image('/content/gdrive/MyDrive/Dyn3DG/data/basketball/ims/0/000000.jpg')
# frame_2 = torchvision.io.read_image('/content/gdrive/MyDrive/Dyn3DG/data/basketball/ims/0/000001.jpg')

def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=1),  # map [0, 1] into [-1, 1]
            T.Resize(size=(520, 960)),
        ]
    )
    batch = transforms(batch)
    return batch

def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()

# # If you can, run this example on a GPU, it will be a lot faster.
device = "cuda" if torch.cuda.is_available() else "cpu"

# img1_batch = preprocess(frame_1).unsqueeze(dim=0).to(device)
# img2_batch = preprocess(frame_2).unsqueeze(dim=0).to(device)

# predicted_flow = of_model(img1_batch, img2_batch)[-1]

# """## train"""

# # Commented out IPython magic to ensure Python compatibility.
# #install the Gaussian Rasterizer
# # %cd diff-gaussian-rasterization-w-depth

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/gdrive/MyDrive/Dyn3DG'

import torch
import os
import json
import copy
import numpy as np
from PIL import Image
from random import randint, seed, sample
from tqdm import tqdm
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2rendervar, params2cpu, save_params
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer
from torchvision.utils import save_image
import faulthandler

faulthandler.enable()


DATASET_PREFIX = dataset_path

def get_dataset(t, md, seq, cameras):
    dataset = []
    for c in cameras:
    #for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md['fn'][t][c]
        im = np.array(copy.deepcopy(Image.open(f"{DATASET_PREFIX}/data/{seq}/ims/{fn}")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255

        seg = np.array(copy.deepcopy(Image.open(f"{DATASET_PREFIX}/data/{seq}/seg/{fn.replace('.jpg', '.png')}"))).astype(np.float32)
        seg = torch.tensor(seg).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
        #epi_col = torch.stack((epi, torch.zeros_like(epi), 1 - epi)) #epi
        gt_flow = np.load(f"{DATASET_PREFIX}/data/{seq}/flow/{fn.replace('.jpg', '_fwd.npz')}")
        gt_flow = torch.from_numpy(gt_flow['flow']).permute(2, 0, 1).cuda()
        dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c, 'gt_flow': gt_flow})
    return dataset


def get_batch(todo_dataset, dataset):
    if "counter" not in get_batch.__dict__: get_batch.counter = 0
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data

def get_curr_and_prev_batch(todo_curr_dataset, todo_prev_dataset, curr_dataset, prev_dataset):
    if not todo_curr_dataset and not todo_prev_dataset:
        todo_curr_dataset = curr_dataset.copy()
        todo_prev_dataset = prev_dataset.copy()

    assert len(todo_curr_dataset) == len(todo_prev_dataset)
    # index = get_batch.counter #randint(0, len(todo_curr_dataset) - 1)
    # curr_data = todo_curr_dataset.pop()
    # prev_data = todo_prev_dataset.pop()
    index = randint(0, len(todo_curr_dataset) - 1)
    curr_data = todo_curr_dataset.pop(index)
    prev_data = todo_prev_dataset.pop(index)
    return curr_data, prev_data


def initialize_params(seq, md, cameras):
    init_pt_cld = np.load(f"{DATASET_PREFIX}/data/{seq}/init_pt_cld.npz")["data"]
    seg = init_pt_cld[:, 6]
    max_cams = 50
    sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    params = {
        'means3D': init_pt_cld[:, :3],
        'rgb_colors': init_pt_cld[:, 3:6],
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),
        'logit_opacities': np.zeros((seg.shape[0], 1)),
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
        'cam_m': np.zeros((max_cams, 3)),
        'cam_c': np.zeros((max_cams, 3)),
        'actual_means2D': torch.zeros(init_pt_cld[:, :3].shape[0], 2)
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_store': {},
                 'first_means2d': {}}
    return params, variables

def update_params_after_first_timestep(params, variables):
    params["actual_means2D"] = torch.nn.Parameter(torch.zeros((params['means3D'].shape[0], 2), requires_grad=True, device="cuda") + 0)
    return params, variables

def init_new_optim_after_first_timestep(params, variables):
    lrs = {
        'means3D': 0.00016 * variables['scene_radius'],
        'rgb_colors': 0.0025,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
        'actual_means2D': 0,
    }
    # just use one random camera
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15, betas=(0.0, 0.0))

def initialize_optimizer(params, variables):
    lrs = {
        'means3D': 0.00016 * variables['scene_radius'],
        'rgb_colors': 0.0025,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
        'actual_means2D': 1e-1,
    }
    # just use one random camera
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
    # return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
    # SGD does not use the second moment
    # return torch.optim.Adak(param_groups, lr=0.00)

def _normalize_flow(flow):
  return (flow - flow.min()) / (flow.max() - flow.min())

def compute_of_seg_loss(of_model, rendered_im, curr_data, prev_data, i, t):
  processed_rendered_im = preprocess(rendered_im).unsqueeze(dim=0).to(device)
  processed_curr_im = preprocess(curr_data['im']).unsqueeze(dim=0).to(device)
  processed_prev_im = preprocess(prev_data['im']).unsqueeze(dim=0).to(device)
  flows_of_rendered = of_model(processed_prev_im, processed_rendered_im)[-1].squeeze()
  flows_of_curr     = of_model(processed_prev_im, processed_curr_im)[-1].squeeze()
  flow_im_rendered  = flow_to_image(flows_of_rendered)
  flow_im_curr      = flow_to_image(flows_of_curr)

  return 0.8 * l1_loss_v1(flow_im_rendered.to(torch.float),
                          flow_im_curr.to(torch.float)) + 0.2 * (1.0 - calc_ssim(flow_im_rendered.to(torch.float), flow_im_curr.to(torch.float)))


def calculate_epe(gt_flow, estimated_flow):
    """
    Calculate the Endpoint Error (EPE) between ground truth and estimated optical flow fields.

    Args:
        gt_flow (torch.Tensor): Ground truth optical flow, shape (batch_size, 2, height, width).
        estimated_flow (torch.Tensor): Estimated optical flow, shape (batch_size, 2, height, width).

    Returns:
        epe (torch.Tensor): EPE between the ground truth and estimated optical flow.
    """
    # Ensure the inputs have the same shape
    assert gt_flow.shape == estimated_flow.shape, "Input shapes must match."

    # Calculate the error vector (du, dv)
    error_vector = gt_flow - estimated_flow

    # Calculate the squared Euclidean distance
    # add eps to stabilize training :)
    squared_distance = torch.sum((error_vector + 1e-6)**2, dim=1)
    # breakpoint()
    # Take the square root to get the EPE
    epe = torch.sqrt(squared_distance)

    # Calculate the mean EPE across the entire batch
    epe = torch.mean(epe)

    return epe

def compute_optical_flow_loss(of_model, rendered_im, curr_data, prev_data):
  processed_rendered_im = preprocess(rendered_im).unsqueeze(dim=0).to(device)
  processed_curr_im = preprocess(curr_data['im']).unsqueeze(dim=0).to(device)
  processed_prev_im = preprocess(prev_data['im']).unsqueeze(dim=0).to(device)
  estimated_flow    = of_model(processed_prev_im, processed_rendered_im)[-1]
  gt_flow           = of_model(processed_prev_im, processed_curr_im)[-1]
  loss_fn           = calculate_epe(gt_flow=gt_flow, estimated_flow=estimated_flow)
  return loss_fn

from torchvision.utils import save_image

last_contrib = None

def get_image_from_means2d(means2d, img_shape):
    means_positions = torch.zeros([1, img_shape[1], img_shape[2]])
    # iterate over means2D and increase pixel value at that position
    for mean in means2d:
        # check if mean is in the image
        if mean[1] < 0 or mean[1] >= img_shape[1] or mean[0] < 0 or mean[0] >= img_shape[2]:
            continue
        means_positions[0, int(mean[1]), int(mean[0])] += 1
    return means_positions

def save_image_from_means2d(means2d, img_shape, save_path):
    means_positions = get_image_from_means2d(means2d, img_shape)
    save_image(means_positions.float() / means_positions.median(), save_path)

def get_loss(params, curr_data, prev_data, variables, is_initial_timestep, i, t, img_number, exp, camera_id):
    losses = {}
    curr_id = curr_data['id']

    segrendervar = params2rendervar(params, curr_id)
    segrendervar['colors_precomp'] = params['seg_colors']
    seg, _, _, _ = Renderer(raster_settings=curr_data['cam'])(**segrendervar)

    rendervar = params2rendervar(params, curr_id)
    rendervar['actual_means2D'].retain_grad()
    im, radius, depth, contrib = Renderer(raster_settings=curr_data['cam'])(**rendervar)

    # breakpoint()
    # if i % 100 == 0:
    #   save_image(im, f'{sandesh_path}/{t}_im_{i}_{img_number}.png')
    # breakpoint()
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    variables['actual_means2D'] = rendervar['actual_means2D']  # Gradient only accum from colour render for densification
    if not is_initial_timestep:
        params['actual_means2D'] = rendervar['actual_means2D']
        visible_ids = contrib.unique().long() # [num_unique]
        visible_means2d = params['actual_means2D'][visible_ids] # [num_unique,2]

    # variables['means2D'] = torch.zeros_like(rendervar['actual_means2D'], requires_grad=True, device="cuda") + 0
     # TODO: needed for accumulation somewhere?


    # contrib: [H,W,1] id of most visible per pixel

   
    #print(curr_data['gt_flow']['flow'].shape, visible_means2d.shape)
    #print(calculate_epe(curr_data['gt_flow']['flow'], visible_means2d))
    
    save_iter = 499 if not is_initial_timestep else 3999
    if i == save_iter:
        save_path = os.path.join(sandesh_path, exp, str(t), str(i))
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        # save_image(seg.float() / 255.0, save_path + "/seg.png")
        print("saving image to " + save_path)
        # save contrib & contrib diff
        save_image(contrib.float() / contrib.max(), save_path + "/contrib.png")
        # save raw image
        save_image(im.float() / (im.median() * 2), save_path + "/im.png")
        # save ground truth
        save_image(curr_data['im'].float() / (curr_data['im'].median() * 2), save_path + "/gt.png")


        ## save means 2d as image
        save_image_from_means2d(rendervar["actual_means2D"], im.shape, save_path + "/means2d.png")

        if  False: # not is_initial_timestep:
            # calculate optical flow from the difference in visible means2d
            print("saving optical flow")
            previous_visible_means2d = variables["prev_means2d_store"][curr_id][visible_ids]
            flow, mask = compute_optical_flow_gaussians(visible_means2d, previous_visible_means2d, im.shape)
            # convert to colored of image
            print("EPE from visible", calculate_epe(curr_data['gt_flow'], flow))
            flow_img = flow_to_image(flow) # return shape is [3hw]            
            save_image(flow_img.float() / 255.0, save_path + "/flow.png")
            # mask out all pixels that are not in the image
            mask = mask.unsqueeze(0).repeat(3, 1, 1)
            masked_flow = flow_img * mask.float()
            save_image(masked_flow.float() / 255.0, save_path + "/flow_masked.png")

            # also calculate optical flow from first means2d
            first_visible_means2d = variables["first_means2d"][curr_id][visible_ids]
            flow_first, mask_first = compute_optical_flow_gaussians(visible_means2d, first_visible_means2d, im.shape)
            print("EPE from first", calculate_epe(curr_data['gt_flow'], flow_first))
            flow_img_first = flow_to_image(flow_first) # return shape is [3hw]
            save_image(flow_img_first.float() / 255.0, save_path + "/flow_first.png")
            # mask out all pixels that are not in the image
            mask_first = mask_first.unsqueeze(0).repeat(3, 1, 1)
            masked_flow_first = flow_img_first * mask_first.float() 
            save_image(masked_flow_first.float() / 255.0, save_path + "/flow_masked_first.png")

            # save flow ground truth
            # flow_image = flow_to_image(curr_data['gt_flow'])
            # save_image(flow_image.float() / 255.0, save_path + "/gt_flow.png")

            flow_object = of.Flow(flow)
            flow_arrow_image = flow_object.visualise_arrows(5)
            save_image(flow_arrow_image.float() / 255.0, save_path + "/flow_arrow.png")

            # save arrow image for ground truth
            flow_object_gt = of.Flow(curr_data['gt_flow'])
            flow_arrow_image_gt = flow_object_gt.visualise_arrows(5)
            save_image(flow_arrow_image_gt.float() / 255.0, save_path + "/flow_arrow_gt.png")

            # save visible means to disk for comparison
            save_image_from_means2d(visible_means2d, im.shape, save_path + "/visible_means2d.png")
            save_image_from_means2d(previous_visible_means2d, im.shape, save_path + "/previous_visible_means2d.png")
            save_image_from_means2d(first_visible_means2d, im.shape, save_path + "/first_visible_means2d.png")
            
    # there is no optical flow at time step 0, therefore we rely on the segmentation masks
    losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))

    USE_OPTICAL_FLOW = False

    if not is_initial_timestep:     
        is_fg = (params['seg_colors'][:, 0] > 0.5).detach() # shape torch.Size([190428]) # [num_points]
        fg_pts = rendervar['means3D'][is_fg]
        fg_rot = rendervar['rotations'][is_fg]
        # breakpoint()

        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
        rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None]
        curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
        losses['rigid'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"],
                                              variables["neighbor_weight"])

        losses['rot'] = weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None],
                                            variables["neighbor_weight"])

        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])

        losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()

        bg_pts = rendervar['means3D'][~is_fg]
        bg_rot = rendervar['rotations'][~is_fg]
        losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])

        losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"])
        if USE_OPTICAL_FLOW:
            # optical flow loss
            previous_visible_means2d = variables["prev_means2d_store"][curr_id][visible_ids]
            # # previous_means2d = variables["prev_means2d_store"][curr_id].to("cuda")
            # # means2d = rendervar['actual_means2D'].to("cuda")
            flow, mask = compute_optical_flow_gaussians(visible_means2d, previous_visible_means2d, im.shape)
            # # flow, mask = compute_optical_flow_gaussians(means2d, previous_means2d, im.shape)

            # # mask out all pixels that are not in the image
            mask = curr_data["seg"][0] # lets use the foreground background segmentation as mask
            flow = flow * mask
            gt_flow = curr_data['gt_flow'] * mask
            flow_loss = calculate_epe(gt_flow, flow)
            # breakpoint()
            losses['optical_flow'] = flow_loss
            # save images of outliers
            if i % 10 == 0: #flow_loss > 0.5
                save_path = os.path.join(sandesh_path, exp, str(t), str(i))
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                print("saving image to " + save_path)
                # save raw image
                save_image(im.float() / (im.median() * 2), save_path + "/im.png")
                # save ground truth
                save_image(curr_data['im'].float() / (curr_data['im'].median() * 2), save_path + "/gt.png")
                # save optical flow
                flow_object = of.Flow(flow)
                flow_arrow_image = flow_object.visualise_arrows(1)
                save_image(flow_arrow_image.float() / 255.0, save_path + "/flow_arrow.png")
                # save arrow image for ground truth
                flow_object_gt = of.Flow(curr_data['gt_flow'])
                flow_arrow_image_gt = flow_object_gt.visualise_arrows(1)
                save_image(flow_arrow_image_gt.float() / 255.0, save_path + "/flow_arrow_gt.png")

                # overlay ground truth and image 
                overlay_image = flow_arrow_image_gt.clone().cuda()
                image = curr_data['im'].float() / (curr_data['im'].median() * 2)
                overlay_image[flow_arrow_image_gt == 255] = (image[flow_arrow_image_gt[0] == 255] * 255.0).byte()

                save_image(overlay_image.float() / 255.0, save_path + "/overlay_gt.png")

                # overlay estimated and image
                overlay_image = flow_arrow_image.clone().cuda()
                overlay_image[flow_arrow_image == 255] = (image[flow_arrow_image[0] == 255] * 255.0).byte()
                save_image(overlay_image.float() / 255.0, save_path + "/overlay_estimated.png")
        else:
            flow_loss = 0.0

       
    loss_weights = {'im': 1.0, 'rigid': 4.0, 'rot': 4.0, 'iso': 2.0, 'floor': 2.0, 'bg': 20.0,
                    'soft_col_cons': 0.01, 'seg': 3.0, 'optical_flow': 0.006}
    #     loss_weights = {'im': 0.0, 'rigid': 0.0, 'rot': 0.0, 'iso': 0.0, 'floor': 0.0, 'bg': 00.0,
    #                 'soft_col_cons': 0.00, 'seg': 0.0, 'optical_flow': 1}
    # else:
    #     loss_weights = {'im': 1.0, 'rigid': 0.0, 'rot': 0.0, 'iso': 0.0, 'floor': 0.0, 'bg': 00.0,
    #                 'soft_col_cons': 0.00, 'seg': 0.0, 'optical_flow': 0.000}    
    
    loss = sum([loss_weights[k] * v for k, v in losses.items()])

    # if is_initial_timestep:
        # save the current means2d for the next timestep
        # variables["first_means2d"][curr_id] = rendervar['actual_means2D'].detach()
        # print(f"storing first means2d, keys are {variables['first_means2d'].keys()}")

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    # variables[f'n_contrib_last_{t}'] = contrib
    # variables[f'means2D_last_{t}'] = rendervar['actual_means2D']

    variables["means2D_store"][curr_id] = rendervar['actual_means2D']
    variables['last_cam_id'] = curr_id
    # breakpoint()
    return loss, variables

def compute_optical_flow_gaussians(visible_means2d: torch.Tensor, visible_means2d_prev: torch.Tensor, img_shape: torch.Tensor) -> torch.Tensor:
    # diff = visible_means2d - visible_means2d_prev # [num_unique, 2]
    diff = visible_means2d_prev - visible_means2d
    optical_flow = torch.zeros([2, img_shape[1], img_shape[2]], device=visible_means2d.device) # [2,H,W]
    mask = torch.zeros([img_shape[1], img_shape[2]], dtype=torch.bool, device=visible_means2d.device)
    for mean, movement in zip(visible_means2d, diff):
        # check if mean is in the image
        if mean[1] < 0 or mean[1] >= img_shape[1] or mean[0] < 0 or mean[0] >= img_shape[2]:
            continue
        optical_flow[0, int(mean[1]), int(mean[0])] += movement[0]
        optical_flow[1, int(mean[1]), int(mean[0])] += movement[1]
        mask[int(mean[1]), int(mean[0])] = True
    return optical_flow, mask

def initialize_per_timestep(params, variables, optimizer):
    pts = params['means3D']
    rot = torch.nn.functional.normalize(params['unnorm_rotations'])
    new_pts = pts + (pts - variables["prev_pts"]) # note(sandesh): i don't understand why they are using this difference here
    # try adding the same for means2d
    detached_means2d = {}
    for camera_id, means2d in variables['means2D_store'].items() :
        # new_means2d[camera_id] = means2d + (means2d - variables["prev_means2d_store"][camera_id])
        detached_means2d[camera_id] = means2d.detach()

    new_rot = torch.nn.functional.normalize(rot + (rot - variables["prev_rot"]))

    is_fg = params['seg_colors'][:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["prev_col"] = params['rgb_colors'].detach()
    variables["prev_pts"] = pts.detach()
    variables["prev_rot"] = rot.detach()
    variables["prev_means2d_store"] = detached_means2d
    # new_means2d = torch.zeros((params['means3D'].shape[0], 2), requires_grad=True, device="cuda") + 0

    # new_params = {'means3D': new_pts, 'unnorm_rotations': new_rot, 'actual_means2D': new_means2d}
    new_params = {'means3D': new_pts, 'unnorm_rotations': new_rot}
    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means3D'][is_fg]
    init_bg_pts = params['means3D'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)
    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means3D'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['unnorm_rotations']).detach()
   
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return variables


def report_progress(params, data, i, progress_bar, camera_id, every_i=100):
    if i % every_i == 0:
        im, _, _, _ = Renderer(raster_settings=data['cam'])(**params2rendervar(params, camera_id=camera_id))
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        psnr = calc_psnr(im, data['im']).mean()
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)


def train(seq, exp):
    if os.path.exists(f"./output/{exp}/{seq}"):
        print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
        return
    print("Preparing training variables")
    md = json.load(open(f"{DATASET_PREFIX}/data/{seq}/train_meta.json", 'r'))  # metadata
    num_timesteps = len(md['fn'])
     ## define camera subset to use
    # lets have reproducible results
    seed(42)
    num_cams = 10
    cameras = sample(range(len(md['fn'][0])), num_cams)
    print("using cameras ", cameras)
    params, variables = initialize_params(seq, md, cameras)
    optimizer = initialize_optimizer(params, variables)
    output_params = []
    prev_data = None  #testing OF idea
    print("Initating Training")
    img_number = 0 #NOTE: TO DELETE

    total_step_count = 0

    for t in range(num_timesteps):
        if t == 1:
            pass
            print("update params and optim to include means2d")
            params, variables = update_params_after_first_timestep(params, variables)
            optimizer = init_new_optim_after_first_timestep(params, variables)
        curr_dataset = get_dataset(t, md, seq, cameras)
        todo_curr_dataset = []
        # todo_prev_dataset = []
        # prev_dataset = None if t == 0 else get_dataset(t - 1, md, seq)
        is_initial_timestep = (t == 0)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)
            # torch.autograd.set_detect_anomaly(True)

        num_iter_per_timestep = 4000 if is_initial_timestep else 500
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        for i in range(num_iter_per_timestep):
            if i == 1:
                print("update params and optim to include means2d")
                params, variables = update_params_after_first_timestep(params, variables)
                optimizer = init_new_optim_after_first_timestep(params, variables)
            # if is_initial_timestep:
            curr_data = get_batch(todo_curr_dataset, curr_dataset)
            prev_data = None
            # else:
            #     prev_data, curr_data = get_curr_and_prev_batch(todo_curr_dataset, todo_prev_dataset, curr_dataset, prev_dataset)
            camera_id = curr_data['id']
            loss, variables = get_loss(params, curr_data, prev_data, variables, is_initial_timestep, i, t, img_number, exp, camera_id)
            writer.add_scalar("Loss/train", loss, total_step_count)
            loss.backward()
            # if t == 1 and i == 20:
            #     breakpoint()
            if i == 1999:
              img_number += 1
            with torch.no_grad():
                report_progress(params, curr_dataset[0], i, progress_bar, camera_id)
                if is_initial_timestep:
                    params, variables = densify(params, variables, optimizer, i)
                clip_grad_norm_(params.values(), max_norm=1.0)
                if params['actual_means2D'].grad is not None and not params['actual_means2D'].grad.isfinite().all():
                    print(f"nan found in means2D tensor: {params['actual_means2D'].grad}")
                    torch.autograd.set_detect_anomaly(True)

                # then use torch.autograd.set_detect_anomaly(True) to debug
                # breakpoint()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_step_count += 1
        writer.flush()
        progress_bar.close()
        output_params.append(params2cpu(params, is_initial_timestep))
        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)
        # save params every iteration
        save_params(output_params, seq, exp)

exp_name = 'exp_of_10_cam_image_reg_of_seperate_fixed'
# "basketball", "boxes", 
for sequence in ["football"]:#, "juggle", "softball", "tennis"]:
    torch.cuda.empty_cache()
    writer = SummaryWriter(log_dir=sandesh_path + f"/log/{exp_name}")
    train(sequence, exp_name)
    torch.cuda.empty_cache()
    #/content/gdrive/MyDrive/Dyn3DG/data/basketball/epipolar_error_png/1/00000.png
writer.close()