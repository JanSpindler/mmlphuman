import os
from os import path
import torch
import numpy as np
import cv2 as cv
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
import imageio.v3 as iio
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from scene.dataset import get_dataset_type, data_to_cam
from scene.gaussian_model import GaussianModel
from scene.net_vis import load_model
from utils.config_utils import Config
from utils.smpl_utils import init_smpl_pose
from utils.eval_utils import eval_images


def load_img_mask(data_dir: str, view_idx: int, pose_idx: int):
    img_path = data_dir + '/images/cam%02d/%08d.jpg' % (view_idx, pose_idx)
    if not os.path.exists(img_path):
        img_path = data_dir + '/images/cam%02d/%08d.png' % (view_idx, pose_idx)

    mask_path = data_dir + '/masks/cam%02d/%08d.jpg' % (view_idx, pose_idx)
    if not os.path.exists(mask_path):
        mask_path = data_dir + '/masks/cam%02d/%08d.png' % (view_idx, pose_idx)

    color_img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    mask_img = cv.imread(mask_path, cv.IMREAD_UNCHANGED)

    # Handle multi-channel mask images
    if mask_img is not None and len(mask_img.shape) == 3:
        if mask_img.shape[2] == 2:
            # Grayscale + alpha: use alpha channel
            mask_img = mask_img[:, :, 1]
        elif mask_img.shape[2] == 4:
            # RGBA: use alpha channel
            mask_img = mask_img[:, :, 3]
        else:
            # RGB: convert to grayscale
            mask_img = cv.cvtColor(mask_img, cv.COLOR_BGR2GRAY)
    
    return color_img, mask_img


@torch.no_grad()
def test(test_run, visualize):
    subject_name = test_run['subject_name']
    ckpt_path = test_run['ckpt_path']
    data_path = test_run['data_path']
    start_frame = test_run['start_frame']
    end_frame = test_run['end_frame']
    views = test_run['views']
    background_color = [0.0, 0.0, 0.0]
    out_dir = test_run.get('out_dir', os.path.join(data_path, f'eval_{subject_name}'))
    device = "cuda"

    # Eval path
    eval_name = f'eval_{subject_name}_frames{start_frame}_{end_frame}_views{"_".join(map(str, views))}.txt'
    eval_path = os.path.join(data_path, eval_name)

    # Init SMPL and load model
    init_smpl_pose()
    gaussians = load_model(ckpt_path)
    gaussians.is_test = True
    gaussians.prepare_test()
    background = torch.as_tensor(np.array(background_color)).float().cuda()

    # Init dataset
    frame_ids = list(range(start_frame, end_frame))
    DatasetType = get_dataset_type(data_path)
    testset = DatasetType(
        datadir=data_path,
        frame_ids=frame_ids,
        cam_ids=views,
        background=np.array(background_color),
        image_scaling=test_run.get('image_scaling', 1.0),
    )

    test_dataloader = DataLoader(
        dataset=testset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f'Initialized dataset with {len(testset)} samples.')

    # Output dirs
    for k in ['gt', 'result', 'mask']:
        os.makedirs(path.join(out_dir, k), exist_ok=True)

    # Clear eval file if it exists
    if os.path.exists(eval_path):
        open(eval_path, 'w').close()

    all_metrics = []
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    for cam in tqdm(test_dataloader):
        cam = data_to_cam(cam, non_blocking=False)
        frame_id = cam['frame_id']
        cam_id = cam.get('cam_id', 0)

        gaussians.smpl_poses = cam['pose']
        gaussians.Th = cam['Th']
        gaussians.Rh = cam['Rh']

        image, alpha, info = gaussians.render(cam, background=background)
        image = (torch.clamp(image, min=0, max=1.0) * 255).byte().contiguous().cpu().numpy()

        image_gt = cam['image']
        image_gt[~cam['mask']] = background
        image_gt = (image_gt * 255).byte().contiguous().cpu().numpy()
        mask = cam['mask'].byte().contiguous().cpu().numpy() * 255

        # Save images
        # iio.imwrite(path.join(out_dir, f'gt/{frame_id:08d}.png'), image_gt)
        # iio.imwrite(path.join(out_dir, f'result/{frame_id:08d}.png'), image)
        # iio.imwrite(path.join(out_dir, f'mask/{frame_id:08d}.png'), mask)

        if visualize:
            cv.imshow('Ground Truth', image_gt)
            cv.imshow('Rendered', image)
            cv.waitKey(1)

        # Compute per-frame metrics
        pred_tensor = torch.from_numpy(image).float().unsqueeze(0).to(device) / 255.0   # (1, H, W, 3)
        ref_tensor = torch.from_numpy(image_gt).float().unsqueeze(0).to(device) / 255.0 # (1, H, W, 3)

        frame_metrics = eval_images(pred_tensor, ref_tensor, fid)
        all_metrics.append(frame_metrics)

        with open(eval_path, 'a') as f:
            f.write(f'cam {cam_id} frame {frame_id}: {frame_metrics}\n')

        torch.cuda.empty_cache()

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    fid_score = fid.compute().item()

    print(f'Average metrics across all frames and views:')
    for key, value in avg_metrics.items():
        print(f'  {key}: {value}')
    print(f'Final FID score: {fid_score}')

    with open(eval_path, 'a') as f:
        f.write(f'Average metrics across all frames and views: {avg_metrics}\n')
        f.write(f'Final FID score across all frames and views: {fid_score}\n')


tests = [
    # # subject00_julian
    # {
    #     "subject_name": "subject00_julian",
    #     "ckpt_path": "./output/subject00_julian/",
    #     "data_path": "./thuman/subject00",
    #     "start_frame": 2000,
    #     "end_frame": 2500,
    #     "views": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    # },
    # {
    #     "subject_name": "subject00_julian",
    #     "ckpt_path": "./output/subject00_julian/",
    #     "data_path": "./thuman/subject00",
    #     "start_frame": 0,
    #     "end_frame": 2000,
    #     "views": [23],
    # },
    # 0206_04
    {
        "subject_name": "0206_04",
        "ckpt_path": "./output/0206_04/",
        "data_path": "./dnarendering/0206_04",
        "start_frame": 180,
        "end_frame": 225,
        "views": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    },
    {
        "subject_name": "0206_04",
        "ckpt_path": "./output/0206_04/",
        "data_path": "./dnarendering/0206_04",
        "start_frame": 0,
        "end_frame": 180,
        "views": [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    }
]


if __name__ == '__main__':
    for test_run in tests:
        test(test_run, False)
