import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from config import load_config
import matplotlib.pyplot as plt
from model.openocc_model import OpenOccModel, qp_to_occ
from model.utils import matrix_to_pose6d, pose6d_to_matrix
from model.utils import coordinates
from dataio.scannet_dataset import ScannetDataset
from dataio.replica_dataset import ReplicaDataset
from dataio.matterport_dataset import MatterportDataset

def main(args):
    config = load_config(scene=args.scene, exp_name=args.exp_name)        
    events_save_dir = os.path.join(config["log_dir"], "events")
    if not os.path.exists(events_save_dir):
        os.makedirs(events_save_dir)
    writer = SummaryWriter(log_dir=events_save_dir)    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config["dataset_type"] == "scannet":
        dataset = ScannetDataset(os.path.join(config["datasets_dir"], args.scene), trainskip=config["trainskip"], openseg=config["openseg"], dino=config["dino"], device=torch.device("cpu"))
    elif config["dataset_type"] == "replica":
        dataset = ReplicaDataset(os.path.join(config["datasets_dir"], args.scene), trainskip=config["trainskip"], openseg=config["openseg"], dino=config["dino"], device=torch.device("cpu"))
    elif config["dataset_type"] == "matterport":
        dataset = MatterportDataset(os.path.join(config["datasets_dir"], args.scene), trainskip=config["trainskip"], openseg=config["openseg"], dino=config["dino"], device=torch.device("cpu"))
    else:
        raise NotImplementedError
    
    model = OpenOccModel(config, device, dataset.get_bounds())
    ray_indices = torch.randperm(len(dataset) * dataset.H * dataset.W)
    
    # Inverse sigma from NeuS paper
    inv_s = nn.parameter.Parameter(torch.tensor(0.3, device=device))
    optimizer = torch.optim.Adam([{"params": model.decoder.parameters(), "lr": config["lr"]["decoder"]},
                                  {"params": model.grid.parameters(), "lr": config["lr"]["features"]},
                                  {"params": model.semantic_grid.parameters(), "lr": config["lr"]["s_features"]},
                                  {"params": inv_s, "lr": config["lr"]["inv_s"]}])
    optimise_poses = config["optimise_poses"]
    poses_mat_init = torch.stack(dataset.c2w_list, dim=0).to(device)
    
    if optimise_poses:
        poses = nn.Parameter(matrix_to_pose6d(poses_mat_init))
        poses_optimizer = torch.optim.Adam([poses], config["lr"]["poses"])

    if args.start_iter > 0:
        state = torch.load(os.path.join(config["checkpoints_dir"], "chkpt_{}".format(args.start_iter)), map_location=device)
        inv_s = state["inv_s"]
        model.load_state_dict(state["model"])
        iteration = state["iteration"]
        optimizer.load_state_dict(state["optimizer"])
        if optimise_poses:
            poses = state["poses"]
            poses_optimizer.load_state_dict(state["poses_optimizer"])
    else:
        center = model.world_dims / 2. + model.volume_origin
        radius = model.world_dims.min() / 4.
        
        # Train occ of a sphere
        for _ in range(500):
            optimizer.zero_grad()
            coords = coordinates(model.voxel_dims[1] - 1, device).float().t()
            pts = (coords + torch.rand_like(coords)) * config["voxel_sizes"][1] + model.volume_origin
            occ, *_ = qp_to_occ(pts.unsqueeze(1), model.volume_origin, model.world_dims, model.grid, model.semantic_grid, model.occ_decoder,
                                concat_qp=config["decoder"]["geometry"]["concat_qp"], rgb_feature_dim=config["rgb_feature_dim"], semantic_feat_dim = config["semantic_feature_dim"])
            occ = occ.squeeze(-1)

            target_occ = (radius < ((center - pts).norm(dim=-1))).float()
            loss = torch.nn.functional.mse_loss(occ, target_occ)
            if loss.item() < 1e-10:
                break            
            loss.backward()
            optimizer.step()
        print("Init loss after geom init (sphere)", loss.item())
    
        # Reset optimizer
        optimizer = torch.optim.Adam([{"params": model.decoder.parameters(), "lr": config["lr"]["decoder"]},
                                      {"params": model.grid.parameters(), "lr": config["lr"]["features"]},
                                      {"params": model.semantic_grid.parameters(), "lr": config["lr"]["s_features"]},
                                      {"params": inv_s, "lr": config["lr"]["inv_s"]}])

    img_stride = dataset.H * dataset.W
    n_batches = ray_indices.shape[0] // config["batch_size"]
    for iteration in trange(args.start_iter + 1, config["iterations"] + 1):
        batch_idx = iteration % n_batches
        ray_ids = ray_indices[(batch_idx * config["batch_size"]):((batch_idx + 1) * config["batch_size"])]
        frame_id = ray_ids.div(img_stride, rounding_mode='floor')
        v = (ray_ids % img_stride).div(dataset.W, rounding_mode='floor')
        u = ray_ids % img_stride % dataset.W
        
        depth = dataset.depth_list[frame_id, v, u].to(device, non_blocking=True)
        rgb = dataset.rgb_list[frame_id, :, v, u].to(device, non_blocking=True)

        if config["openseg"]:
            if config["dataset_type"] == "matterport":
                feat_2d = torch.zeros([config["batch_size"], dataset.feat_dim])
                feat_2d_id = frame_id // 100
                feat_2d_index = frame_id % 100
                for i in range(dataset.feats_list_len):
                    feat_2d[feat_2d_id == i] = dataset.feats_list[i][feat_2d_index[feat_2d_id == i],:, v[feat_2d_id == i], u[feat_2d_id == i]].float()
                feat_2d = feat_2d.to(device, non_blocking=True)
            else:
                feat_2d = dataset.feats_list[frame_id, :, v, u].to(device, non_blocking=True)
        else:
            feat_2d = None

        if config["dino"]:
            img_scale = (
                dataset.dino_list.shape[2] / dataset.H,
                dataset.dino_list.shape[3] / dataset.W,
            )
            v_new, u_new = (v * img_scale[0]).long(), (u * img_scale[1]).long()
            dino_feat_2d = dataset.dino_list[frame_id, :, v_new, u_new].to(device, non_blocking=True)
        else:
            dino_feat_2d = None
        
        fx, fy = dataset.K_list[frame_id, 0, 0], dataset.K_list[frame_id, 1, 1]
        cx, cy = dataset.K_list[frame_id, 0, 2], dataset.K_list[frame_id, 1, 2]

        if config["dataset_type"] == "scannet" or config["dataset_type"] == "matterport":
            rays_d_cam = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(fx)], dim=-1).to(device)
        else:
            rays_d_cam = torch.stack([(u - cx) / fx, -(v - cy) / fy, -torch.ones_like(fy)], dim=-1).to(device)
        
        if optimise_poses:
            batch_poses = poses[frame_id]
            c2w = pose6d_to_matrix(batch_poses)
        else:
            c2w = poses_mat_init[frame_id]
        
        rays_o = c2w[:,:3,3]
        rays_d = torch.bmm(c2w[:, :3, :3], rays_d_cam[..., None]).squeeze()
        
        if config["depth_mask"]:
            mask = (depth > 0) & (depth < 8)
            depth = depth[mask]
            rays_o = rays_o[mask]
            rays_d = rays_d[mask]
            rgb = rgb[mask]
            if config["openseg"]:
                feat_2d = feat_2d[mask]
            if config["dino"]:
                dino_feat_2d = dino_feat_2d[mask]

        ret = model(rays_o, rays_d, rgb, depth, feat_2d, dino_feat_2d, inv_s=torch.exp(10. * inv_s),
                    smoothness_std=config["smoothness_std"], iter=iteration)

        loss = config["rgb_weight"] * ret["rgb_loss"] +\
               config["depth_weight"] * ret["depth_loss"] +\
               config["fs_weight"] * ret["fs_loss"] +\
               config["occ_weight"] * ret["occ_loss"] +\
               config["semantic_weight"] * ret["semantic_loss"] +\
               config["dino_weight"] * ret["dino_loss"] +\
               config["entroy_weight"] * ret["entropy_loss"]        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.grid.parameters(), 1.)
        torch.nn.utils.clip_grad_norm_(model.semantic_grid.parameters(), 1.)
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.)
        
        optimizer.step()
        optimizer.zero_grad()
        
        if optimise_poses:
            if iteration > 100:
                if iteration % 3 == 0:
                    poses_optimizer.step()
                    poses_optimizer.zero_grad()
            else:
                poses_optimizer.zero_grad()

        writer.add_scalar('depth', ret["depth_loss"].item(), iteration)
        writer.add_scalar('rgb', ret["rgb_loss"].item(), iteration)
        writer.add_scalar('sem', ret["semantic_loss"].item(), iteration)
        writer.add_scalar('dino', ret["dino_loss"].item(), iteration)
        writer.add_scalar('entropy', ret["entropy_loss"].item(), iteration)
        writer.add_scalar('fs', ret["fs_loss"].item(), iteration)
        writer.add_scalar('occ', ret["occ_loss"].item(), iteration)
        writer.add_scalar('psnr', ret["psnr"].item(), iteration)
        
        if iteration % args.i_print == 0:
            tqdm.write("Iter: {}, PSNR: {:6f}, RGB Loss: {:6f}, Depth Loss: {:6f}, occ Loss: {:6f}, FS Loss: {:6f}, SeM Loss:{:6f}, Dino Loss:{:6f}, Entropy Loss:{:6f}"
                       .format(iteration,ret["psnr"].item(), ret["rgb_loss"].item(), 
                               ret["depth_loss"].item(),ret["occ_loss"].item(),
                               ret["fs_loss"].item(),ret["semantic_loss"].item(), 
                               ret["dino_loss"].item(), ret["entropy_loss"].item()))
        
        if iteration % args.i_save == 0:
            state = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'iteration': iteration,
                     'poses': poses if 'poses' in locals() else None,
                     'poses_optimizer': poses_optimizer.state_dict() if 'poses_optimizer' in locals() else None,
                     'inv_s': inv_s}
            torch.save(state, os.path.join(config["checkpoints_dir"], "chkpt_{}".format(iteration)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="mp_label35_test")
    parser.add_argument('--scene', type=str, default="2t7WUuJeko7")
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--i_print', type=int, default=20)
    parser.add_argument('--i_save', type=int, default=10000)
    args = parser.parse_args()
    main(args)
