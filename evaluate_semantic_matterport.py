import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import glob

from dataio.matterport_dataset import MatterportDataset
from model.openocc_model import OpenOccModel
from config import load_config
import marching_cubes as mcubes
from tools.text_clip import precompute_text_related_properties, convert_labels_with_palette

import open3d as o3d
from tools.metric import evaluate

def batchify_vol_semantic(fn, chunk, dim=[768]):
    if chunk is None:
        return fn
    def ret(model, coords, **kwargs):
        full_val = torch.empty([coords.shape[0]] + dim, device=coords.device)
        for i in range(0, coords.shape[0], chunk):
            val = fn(model, coords[i:i+chunk,:].contiguous(), **kwargs)
            full_val[i:i+chunk,:] = val
        return full_val
    return ret

def query_semantic(model, coords, volume_origin, world_dim, use_normals=False,
              use_view_dirs=False, concat_qp_occ=False,
              concat_qp_rgb=False, use_dot_prod=False, rgb_feature_dim=[]):
    coords = coords.float()
    vertices_query = 2. * (coords.view(1, 1, 1, -1, 3) - volume_origin) / world_dim - 1.
    semantic_feats = model.semantic_grid(vertices_query[..., [2, 1, 0]], concat=False)
    semantic_feats = torch.cat(semantic_feats, dim=1).squeeze(0).squeeze(1).squeeze(1).t()
        
    semantic = model.semantic_decoder(semantic_feats)

    semantic = semantic / (torch.linalg.norm(semantic, dim=-1, keepdim=True) + 1e-5)

    return semantic

def main(args):
    config = load_config(scene=args.scene, exp_name=args.exp_name, use_config_snapshot=True)
    suffix = config["iterations"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    work_dir = os.path.join(config["log_dir"], "eval/{}".format(suffix))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    dataset = MatterportDataset(os.path.join(config["datasets_dir"], args.scene),load=False, trainskip=config["trainskip"], openseg=False, dino=False, device=torch.device("cpu"))
    model = OpenOccModel(config, device, dataset.get_bounds())
    state = torch.load(os.path.join(config["checkpoints_dir"], "chkpt_{}".format(suffix)), map_location=device)
    model.load_state_dict(state["model"])

    gt_mesh_dirs = sorted(glob.glob(os.path.join('script/matterport_3d/label35',  args.scene+ "*" + ".pth")))  
    mean_ious = []
    mean_accs = []
    for gt_mesh_dir in gt_mesh_dirs:
        region_name = gt_mesh_dir.split("/")[-1].split(".")[0]
        locs_in, feats_in, labels_in = torch.load(gt_mesh_dir) 

        labels_gt = labels_in.astype(np.uint8)
        labels_gt = torch.from_numpy(labels_gt).long()
        coords = torch.from_numpy(locs_in)
        color_gt = torch.from_numpy(feats_in).float() / 255.0

        semantic = batchify_vol_semantic(query_semantic, 1024)(model, torch.tensor(coords, device=model.device),
                                            volume_origin = model.volume_origin, 
                                            world_dim = model.world_dims,
                                            use_normals=config["decoder"]["radiance"]["use_normals"],
                                            use_view_dirs=config["decoder"]["radiance"]["use_view_dirs"],
                                            concat_qp_occ=config["decoder"]["geometry"]["concat_qp"],
                                            concat_qp_rgb=config["decoder"]["radiance"]["concat_qp"],
                                            use_dot_prod=config["decoder"]["radiance"]["use_dot_prod"],
                                            rgb_feature_dim=config["rgb_feature_dim"])

        labelset_name='mp_label35'
        text_features, labelset, mapper, palette = \
            precompute_text_related_properties(labelset_name)
        
        rendered = semantic.half() @ text_features.t()
        logits_rendered = torch.max(rendered, 1)[1].detach().cpu()
        rendered_label_color = convert_labels_with_palette(logits_rendered.numpy(), palette)

        if args.save_pc:
            coords = coords.detach().cpu().numpy()
            device = o3d.core.Device("cuda:0")
            cloud = o3d.t.geometry.PointCloud(device)
            cloud.point["positions"] = o3d.core.Tensor(coords)
            cloud.point["colors"] = o3d.core.Tensor(color_gt.numpy())
            o3d.t.io.write_point_cloud(os.path.join(work_dir,"pc_label_gt_{}.ply".format(region_name)), cloud)

            cloud.point["colors"] = o3d.core.Tensor(rendered_label_color)
            o3d.t.io.write_point_cloud(os.path.join(work_dir,"pc_label_rendered_{}.ply".format(region_name)), cloud)

        # these labels were marked as objects in gt, so we do not evaluate.
        mask = (logits_rendered >= 27) & (logits_rendered<=34) 
        logits_rendered[mask] = 255
        
        mean_iou, mean_acc = evaluate(pred_ids = logits_rendered, gt_ids = labels_gt, dataset='mp_label35')
        mean_ious.append(mean_iou)
        mean_accs.append(mean_acc)
    
    print("Final Mean_iou:", sum(mean_ious)/len(mean_ious))
    print("Final Mean_acc:", sum(mean_accs)/len(mean_accs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="mp_label35_test")
    parser.add_argument("--scene", type=str, default="2t7WUuJeko7")
    parser.add_argument("--save_pc", default=True) 
    args = parser.parse_args()

    scene_names = ["2t7WUuJeko7", "jh4fc5c5qoQ","zsNo4HB9uLZ"]
    for scene_name in scene_names:
        args.scene = scene_name
        main(args)