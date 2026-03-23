import os
import torch
import numpy as np
import cv2 as cv
from tqdm import tqdm
from torch.utils.data import DataLoader

from scene.dataset import get_dataset_type, data_to_cam
from scene.net_vis import load_model
from utils.smpl_utils import init_smpl_pose


@torch.no_grad()
def render_cam23():
    # --- Configuration ---
    ckpt_path = "./output/subject00_julian/"  # Change to your checkpoint path
    data_path = "./thuman/subject00"          # Change to your data path
    start_frame = 0
    num_frames = 10
    end_frame = start_frame + num_frames
    camera_id = 23
    background_color = [0.0, 0.0, 0.0]
    out_dir = "./output_cam23"

    device = "cuda"

    # --- Output directories ---
    os.makedirs(os.path.join(out_dir, "rendered"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "reference"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "side_by_side"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "mask"), exist_ok=True)

    # --- Init SMPL and load model ---
    init_smpl_pose()
    gaussians = load_model(ckpt_path)
    gaussians.is_test = True
    gaussians.prepare_test()
    background = torch.as_tensor(np.array(background_color)).float().cuda()

    # --- Init dataset ---
    frame_ids = list(range(start_frame, end_frame))
    DatasetType = get_dataset_type(data_path)
    testset = DatasetType(
        datadir=data_path,
        frame_ids=frame_ids,
        cam_ids=[camera_id],
        background=np.array(background_color),
        image_scaling=1.0,
    )

    test_dataloader = DataLoader(
        dataset=testset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Rendering {num_frames} frames for camera {camera_id}...")
    print(f"Output directory: {out_dir}")

    for i, cam in enumerate(tqdm(test_dataloader)):
        cam = data_to_cam(cam, non_blocking=False)
        frame_id = cam['frame_id']

        # Set SMPL pose
        gaussians.smpl_poses = cam['pose']
        gaussians.Th = cam['Th']
        gaussians.Rh = cam['Rh']

        # Render
        image, alpha, info = gaussians.render(cam, background=background)
        rendered = (torch.clamp(image, min=0, max=1.0) * 255).byte().cpu().numpy()

        # Masked reference image
        image_gt = cam['image'].clone()
        image_gt[~cam['mask']] = torch.tensor(background_color, device=image_gt.device)
        reference = (image_gt * 255).byte().cpu().numpy()
        mask = (cam['mask'].float() * 255).byte().cpu().numpy()

        # Save rendered image (RGB -> BGR for cv2)
        rendered_bgr = cv.cvtColor(rendered, cv.COLOR_RGB2BGR)
        reference_bgr = cv.cvtColor(reference, cv.COLOR_RGB2BGR)

        rendered_path = os.path.join(out_dir, "rendered", f"frame_{frame_id:08d}_cam{camera_id:02d}.png")
        reference_path = os.path.join(out_dir, "reference", f"frame_{frame_id:08d}_cam{camera_id:02d}.png")
        side_by_side_path = os.path.join(out_dir, "side_by_side", f"frame_{frame_id:08d}_cam{camera_id:02d}.png")
        mask_path = os.path.join(out_dir, "mask", f"frame_{frame_id:08d}_cam{camera_id:02d}.png")

        cv.imwrite(rendered_path, rendered_bgr)
        cv.imwrite(reference_path, reference_bgr)
        cv.imwrite(mask_path, mask)

        # Side-by-side comparison
        side_by_side = np.concatenate([reference_bgr, rendered_bgr], axis=1)
        cv.imwrite(side_by_side_path, side_by_side)

        torch.cuda.empty_cache()

    print(f"\nDone! Images saved to:")
    print(f"  Rendered:     {out_dir}/rendered/")
    print(f"  Reference:    {out_dir}/reference/")
    print(f"  Side-by-side: {out_dir}/side_by_side/")
    print(f"  Mask:         {out_dir}/mask/")


if __name__ == "__main__":
    render_cam23()