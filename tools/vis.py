import argparse, torch, os
import numpy as np
from mayavi import mlab
import mayavi

import torch.nn.functional as F
from dataio.scannet_dataset import ScannetDataset
from model.openocc_model import OpenOccModel, compute_grads
from model.openocc_model import qp_to_occ
from config import load_config
from tools.text_clip import precompute_text_related_properties, convert_labels_with_palette_vis

def batchify_vol(fn, chunk, dim=[]):
    if chunk is None:
        return fn
    def ret(model, coords, volume_origin, volume_dim, **kwargs):
        full_val = torch.empty(list(coords.shape[0:3]) + dim, device=coords.device)
        for i in range(0, coords.shape[2], chunk):
            val = fn(model, coords[:,:,i:i+chunk,:].contiguous(), volume_origin, volume_dim, **kwargs)
            full_val[:,:,i:i+chunk] = val
        return full_val
    return ret

def batchify_semantic(fn, chunk, dim=[768]):
    if chunk is None:
        return fn
    def ret(model, coords, volume_origin, volume_dim, world_dim, **kwargs):
        full_val = torch.empty(list(coords.shape[0:3]) + dim, device=coords.device)
        for i in range(0, coords.shape[2], chunk):
            val = fn(model, coords[:,:,i:i+chunk,:].contiguous(), volume_origin, volume_dim, world_dim, **kwargs)
            full_val[:,:,i:i+chunk,:] = val
        return full_val
    return ret

def get_grid_coords(dims, voxel_origin, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """
    voxel_new_dims = [(i-1) * (voxel_origin / torch.tensor(resolution)).int() + 1 for i in dims]

    g_xx = np.arange(0, voxel_new_dims[0]) 
    g_yy = np.arange(0, voxel_new_dims[1])
    g_zz = np.arange(0, voxel_new_dims[2])

    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.stack([xx, yy, zz], axis=-1)
    coords_grid = coords_grid.astype(np.float32)
    resolution = resolution * np.ones_like(coords_grid)

    coords_grid = (coords_grid * resolution) + resolution / 2
    unit_grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, voxel_new_dims[0]),
                                           torch.linspace(-1, 1, voxel_new_dims[1]),
                                           torch.linspace(-1, 1, voxel_new_dims[2])))
    return coords_grid, voxel_new_dims, unit_grid


def query_semantic(model, pts, volume_origin, volume_dim, world_dim):
    pts_norm = 2. * (pts.to(volume_origin) - volume_origin[None, None, None, :]) / world_dim[None, None, None, :].to(
        volume_origin) - 1.
    pts_norm = pts_norm.unsqueeze(0)

    semantic_feats = model.semantic_grid(pts_norm[..., [2, 1, 0]], concat=False)
    semantic_feats = torch.cat(semantic_feats, dim=1).squeeze(0).permute(1, 2, 3, 0)
    semantic = model.semantic_decoder(semantic_feats)
    semantic = semantic / (torch.linalg.norm(semantic, dim=-1, keepdim=True) + 1e-5)
    return semantic

def query_occ(model, pts, volume_origin, world_dim, concat_qp=False, rgb_feature_dim=[]):
    pts_norm = 2. * (pts.to(volume_origin) - volume_origin[None, None, None, :]) / world_dim[None, None, None, :].to(volume_origin) - 1.
    mask = (pts_norm.abs() <= 1.).all(dim=-1)
    pts_norm = pts_norm[mask]

    pts_norm_mask = torch.zeros(list(mask.shape)+[3], device=pts_norm.device)
    pts_norm_mask[mask] = pts_norm
    pts_norm_mask = pts_norm_mask.unsqueeze(0)
    mlvl_feats = model.grid(pts_norm_mask[..., [2, 1, 0]], concat=False)

    occ_feats = list(map(lambda feat_pts, rgb_dim: feat_pts[:, :-rgb_dim, ...] if rgb_dim > 0 else feat_pts,
                         mlvl_feats, rgb_feature_dim))
    if concat_qp:
        occ_feats.append(pts.permute(0, 4, 1, 2, 3))
    occ = model.decoder.geometry_net(torch.cat(occ_feats, dim=1).squeeze(0).permute(1, 2, 3, 0))[..., 0]

    occ_mask = torch.zeros_like(occ)
    occ = occ[mask]
    occ_mask[mask] = occ
    return occ_mask

def query_rgb(model, coords, volume_origin, world_dim, use_normals=False,
              use_view_dirs=False, concat_qp_occ=False,
              concat_qp_rgb=False, use_dot_prod=False, rgb_feature_dim=[]):
    with torch.enable_grad():
        coords = coords.float().requires_grad_(True)
        vertices_query = 2. * (coords.view(1, 1, 1, -1, 3) - volume_origin) / world_dim - 1.
        mlvl_feats = model.grid(vertices_query[..., [2, 1, 0]], concat=False)

        occ_feats = list(map(lambda feat_pts, rgb_dim: feat_pts[:, :-rgb_dim, ...] if rgb_dim > 0 else feat_pts,
                             mlvl_feats, rgb_feature_dim))
        if concat_qp_occ:
            occ_feats.append(vertices_query.permute(0, 4, 1, 2, 3))

        occ = model.decoder.geometry_net(torch.cat(occ_feats, dim=1).squeeze(0).permute(1, 2, 3, 0))[..., 0]
        grads = F.normalize(compute_grads(occ, coords), dim=-1).squeeze(0).detach()

    rgb_feats = map(lambda feat_pts, rgb_dim: feat_pts[:, -rgb_dim:, ...] if rgb_dim > 0 else None,
                    mlvl_feats, rgb_feature_dim)
    rgb_feats = list(filter(lambda x: x is not None, rgb_feats))
    rgb_feats = [torch.cat(rgb_feats, dim=1).squeeze(0).squeeze(1).squeeze(1).t()]

    if use_view_dirs:
        rgb_feats.append(-grads)
    if use_normals:
        rgb_feats.append(grads)
    if concat_qp_rgb:
        rgb_feats.append(vertices_query.squeeze())
    if use_dot_prod:
        rgb_feats.append(-torch.ones_like(grads[..., :1]))

    rgb = torch.sigmoid(model.decoder.radiance_net(torch.cat(rgb_feats, dim=-1)))
    return rgb, grads, coords

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath('.'))
    device = torch.device('cuda:0')

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py_config', default='config/occupancy.py')
    parser.add_argument("--exp_name", type=str, default="occ_seg_mask_ours31_entropy2")
    parser.add_argument('--scene_name', type=str, default='scene0000_00', help='names of scenes to visualize')
    parser.add_argument('--scene_idx', type=int, default=0, nargs='+', help='idx of scenes to visualize')
    args = parser.parse_args()
    print(args)

    config = load_config(scene=args.scene_name, exp_name=args.exp_name, use_config_snapshot=True)
    suffix = config["iterations"]
    vis_occ_dir = os.path.join(config["log_dir"], "occ/{}".format(suffix))
    if not os.path.exists(vis_occ_dir):
        os.makedirs(vis_occ_dir)


    dataset = ScannetDataset(os.path.join(config["datasets_dir"], args.scene_name), load=False,
                                 trainskip=config["trainskip"], device=torch.device("cpu"))
    model = OpenOccModel(config, device, dataset.get_bounds())
    state = torch.load(os.path.join(config["checkpoints_dir"], "chkpt_{}".format(suffix)), map_location=device)
    model.load_state_dict(state["model"])

    vis_resolution = 0.05
    with torch.no_grad():
        _, _, nx, ny, nz = model.grid.volumes[3].shape
        if config["voxel_sizes"][0] > vis_resolution:
            print("Error!!! The resample voxel size is less than the lowest resolution grid size")
        grid_coords, voxel_new_dim, unit_grid = get_grid_coords( [nx, ny, nz], config["voxel_sizes"][3], vis_resolution)
        grid_coords = grid_coords + np.array(model.volume_origin.cpu(), dtype=np.float32)[None, None, None, :]

        grid_coords = torch.from_numpy(grid_coords)
        occ = batchify_vol(query_occ, 1)(model, grid_coords, model.volume_origin, model.world_dims,
                                         concat_qp=config["decoder"]["geometry"]["concat_qp"],
                                         rgb_feature_dim=config["rgb_feature_dim"])

        occ_mask = occ.ge(0.5)
        semantic = batchify_semantic(query_semantic, 1)(model, grid_coords, model.volume_origin,
                                  voxel_new_dim, model.world_dims,
                                  use_normals=config["decoder"]["radiance"]["use_normals"],
                                  use_view_dirs=config["decoder"]["radiance"]["use_view_dirs"],
                                  concat_qp_occ=config["decoder"]["geometry"]["concat_qp"],
                                  concat_qp_rgb=config["decoder"]["radiance"]["concat_qp"],
                                  use_dot_prod=config["decoder"]["radiance"]["use_dot_prod"],
                                  rgb_feature_dim=config["rgb_feature_dim"])

        semantic_with_occ = semantic[occ_mask, :]
        fov_voxels = grid_coords[occ_mask, :]

        labelset_name = 'ours_30'
        text_features, labelset, mapper, palette = \
            precompute_text_related_properties(labelset_name)

        rendered = semantic_with_occ.to(text_features).half() @ text_features.t()
        logits_rendered = torch.max(rendered, 1)[1].detach().cpu()
        
        z_filter = True
        if z_filter:
            z_mask = fov_voxels[:, 2] < (torch.max(fov_voxels[:, 2]) - vis_resolution * 6)
            xx = fov_voxels[:, 0][z_mask]
            yy = fov_voxels[:, 1][z_mask]
            zz = fov_voxels[:, 2][z_mask]
            labels = logits_rendered[z_mask].numpy()
        else:
            xx = fov_voxels[:, 0]
            yy = fov_voxels[:, 1]
            zz = fov_voxels[:, 2]
            labels = logits_rendered.numpy()

        figure = mlab.figure(size=(1280, 720), bgcolor=(1, 1, 1))
        label_voxels = logits_rendered.unsqueeze(-1).numpy()
        fov_voxels = fov_voxels.numpy()

        plt_plot_fov = mlab.points3d(
            xx,
            yy,
            zz,
            labels,
            colormap="viridis",
            scale_factor=0.95 * vis_resolution,
            mode="cube",
            opacity=1.0
        )

        colors = convert_labels_with_palette_vis(logits_rendered.numpy(), palette)
        colors = colors.astype(np.uint8)
        plt_plot_fov.glyph.scale_mode = "scale_by_vector"
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
        mlab.draw()
        mlab.show()
