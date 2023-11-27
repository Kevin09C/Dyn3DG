import torch
import os
import json
import copy
import numpy as np
from PIL import Image
from random import randint
from tqdm import tqdm
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2rendervar, params2cpu, save_params
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer
from torchvision.utils import save_image

def get_dataset(t, md, seq):
    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md['fn'][t][c]
        im = np.array(copy.deepcopy(Image.open(f"./data/{seq}/ims/{fn}")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        dataset.append({'cam': cam, 'im': im,'id': c})
    return dataset



def calc_metrics(rendervar, data, params, t, seq, i):
    im, _, _, = Renderer(raster_settings=data['cam'])(**rendervar)
    curr_id = data['id']
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
    save_image(im, f'./data/{seq}/rendered/test_time_img/timestep_{t}_img_{i}.png')
    psnr = calc_psnr(im, data['im']).mean()
    ssim = calc_ssim(im, data['im']).mean()
    return psnr, ssim


def evaluate(seq, exp):
    md = json.load(open(f"./data/{seq}/test_meta.json", 'r'))  # metadata
    num_timesteps = len(md['fn'])
    params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    psnr_arr = []
    ssim_arr = []
    with torch.no_grad():
        for t in range(num_timesteps):
            dataset = get_dataset(t, md, seq)
            rendervar = {
                'means3D': params['means3D'][t],
                'colors_precomp': params['rgb_colors'][t],
                'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
                'opacities': torch.sigmoid(params['logit_opacities']),
                'scales': torch.exp(params['log_scales']),
                'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
            }
            for i in range(len(dataset)):
                curr_data = dataset[i]
                psnr, ssim = calc_metrics(rendervar, curr_data, params, t, seq, i)
                psnr_arr.append(psnr)
                ssim_arr.append(ssim)
        avg_psnr = sum(psnr_arr)/len(psnr_arr)
        avg_ssim = sum(ssim_arr)/len(ssim_arr)
        print(f"Sequence: {seq} \t\t PSNR: {avg_psnr:.{7}} \t SSIM: {avg_ssim:.{7}}")

if __name__ == "__main__":
    exp_name = "exp1"
    for sequence in ["basketball", "boxes", "football", "juggle", "softball", "tennis"]:
        evaluate(sequence, exp_name)
