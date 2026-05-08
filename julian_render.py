import os
from os import path
import torch
import numpy as np
import cv2 as cv
from tqdm import tqdm
import imageio.v3 as iio
from omegaconf import OmegaConf
import argparse

from scene.dataset import ThumanDataset, data_to_cam
from scene.gaussian_model import GaussianModel
from scene.net_vis import load_model
from utils.config_utils import Config
from utils.smpl_utils import init_smpl_pose
from torch.utils.data import DataLoader


@torch.inference_mode()
def render(test_run):
    subject_name = test_run['subject_name']
    ckpt_path = test_run['ckpt_path']
    data_path = test_run['data_path']
    start_frame = test_run['start_frame']
    end_frame = test_run['end_frame']
    views = test_run['views']
    background_color = [1.0, 1.0, 1.0]
    out_dir = test_run.get(
        'out_dir',
        os.path.join('renders', subject_name, f'frames{start_frame}_{end_frame}')
    )
    device = "cuda"

    init_smpl_pose()
    gaussians = load_model(ckpt_path)
    gaussians.is_test = True
    gaussians.prepare_test()
    background = torch.as_tensor(np.array(background_color)).float().cuda()

    frame_ids = list(range(start_frame, end_frame))
    testset = ThumanDataset(
        datadir=data_path,
        frame_ids=frame_ids,
        cam_ids=views,
        background=np.array(background_color),
        image_scaling=test_run.get('image_scaling', 1.0),
    )

    print(f'Initialized dataset with {len(testset)} samples.')

    os.makedirs(path.join(out_dir, "result"), exist_ok=True)

    dataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    for cam in tqdm(dataloader, desc=f'Rendering {subject_name} frames {start_frame}-{end_frame}'):
        cam = data_to_cam(cam, non_blocking=True)
        frame_id = cam['frame_id']
        cam_id = cam.get('cam_id', 0)

        gaussians.smpl_poses = cam['pose']
        gaussians.Th = cam['Th']
        gaussians.Rh = cam['Rh']

        image, alpha, info = gaussians.render(cam, background=background)
        del alpha, info

        image_np = (torch.clamp(image, min=0, max=1.0) * 255).byte().contiguous().cpu().numpy()

        image_gt = cam['image'].clone()
        image_gt[~cam['mask']] = background
        image_gt_np = (image_gt * 255).byte().contiguous().cpu().numpy()

        this_dir = path.join(out_dir, "result", f"cam{cam_id:02d}")
        os.makedirs(this_dir, exist_ok=True)
        iio.imwrite(path.join(this_dir, f"{frame_id:08d}.png"), image_np)
        # iio.imwrite(path.join(out_dir, 'gt', fname), image_gt_np)

        del image, image_np, image_gt, image_gt_np, cam
        # torch.cuda.empty_cache()

    print(f'Saved renders to {out_dir}')


tests = [
    # subject00
    {
        "subject_name": "subject00",
        "ckpt_path": "./output/subject00/",
        "data_path": "./thuman/subject00",
        "start_frame": 0,
        "end_frame": 2500,
        "views": list(range(24)),
    },
    # subject01
    {
        "subject_name": "subject01",
        "ckpt_path": "./output/subject01/",
        "data_path": "./thuman/subject01",
        "start_frame": 0,
        "end_frame": 2500,
        "views": list(range(24)),
    },
    # subject02
    {
        "subject_name": "subject02",
        "ckpt_path": "./output/subject02/",
        "data_path": "./thuman/subject02",
        "start_frame": 0,
        "end_frame": 2500,
        "views": list(range(24)),
    },
    # 0165_08
    {
        "subject_name": "0165_08",
        "ckpt_path": "./output/0165_08/",
        "data_path": "./dnarendering/0165_08",
        "start_frame": 0,
        "end_frame": 225,
        "views": list(range(60)),
    },
    # 0166_04
    {
        "subject_name": "0166_04",
        "ckpt_path": "./output/0166_04/",
        "data_path": "./dnarendering/0166_04",
        "start_frame": 0,
        "end_frame": 225,
        "views": list(range(60)),
    },
    # 0206_04
    {
        "subject_name": "0206_04",
        "ckpt_path": "./output/0206_04/",
        "data_path": "./dnarendering/0206_04",
        "start_frame": 0,
        "end_frame": 225,
        "views": list(range(60)),
    }
]


if __name__ == '__main__':
    print(f'Found {len(tests)} test(s). Running them sequentially...')
    for test_run in tests:
        render(test_run)
