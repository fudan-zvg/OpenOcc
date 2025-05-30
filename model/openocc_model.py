import torch
from torch import nn
import torch.nn.functional as F
from model.decoder import NeRFDecoder
from model.multi_grid import MultiGrid
from model.utils import compute_world_dims

BCELoss = nn.BCELoss()
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x)

class OpenOccModel(torch.nn.Module):
    def __init__(self,
                 config,
                 device,
                 bounds,
                 margin=0.1,
                 ):
        super(OpenOccModel, self).__init__()
        self.device = device
        
        world_dims, volume_origin, voxel_dims = compute_world_dims(bounds,
                                                                   config["voxel_sizes"],
                                                                   len(config["voxel_sizes"]),
                                                                   margin=margin,
                                                                   device=device)
        self.world_dims = world_dims
        self.volume_origin = volume_origin
        self.voxel_dims = voxel_dims
        
        grid_dim = (torch.tensor(config["occ_feature_dim"]) + torch.tensor(config["rgb_feature_dim"])).tolist()
        self.grid = MultiGrid(voxel_dims, grid_dim).to(device)
        semantic_dim = (torch.tensor(config["semantic_feature_dim"]))
        self.semantic_grid = MultiGrid(voxel_dims, semantic_dim).to(device)
        
        self.decoder = NeRFDecoder(config["decoder"]["geometry"],
                                   config["decoder"]["radiance"],
                                   config["decoder"]["semantic"],
                                   config["decoder"]["dino"],
                                   occ_feat_dim=sum(config["occ_feature_dim"]),
                                   rgb_feat_dim=sum(config["rgb_feature_dim"]),
                                   semantic_feat_dim=sum(config["semantic_feature_dim"]),
                                   dino_feat_dim=sum(config["semantic_feature_dim"])).to(device)
        self.occ_decoder = batchify(self.decoder.geometry_net, max_chunk=None)
        self.rgb_decoder = batchify(self.decoder.radiance_net, max_chunk=None)
        self.semantic_decoder = batchify(self.decoder.semantic_net, max_chunk=None)
        self.dino_decoder = batchify(self.decoder.dino_net, max_chunk=None)
        self.config = config

    def forward(self, rays_o, rays_d, target_rgb_select, target_depth_select, feat_2d, dino_feat_2d, inv_s=None, smoothness_std=0., iter=0):       
        rend_dict = render_rays(self.occ_decoder,
                                self.rgb_decoder,
                                self.semantic_decoder,
                                self.dino_decoder,
                                self.grid,
                                self.semantic_grid,
                                self.volume_origin,
                                self.world_dims,
                                self.config["voxel_sizes"],
                                rays_o,
                                rays_d,
                                near=self.config["near"],
                                far=self.config["far"],
                                n_samples=self.config["n_samples"],
                                depth_gt=target_depth_select,
                                n_importance=self.config["n_importance"],
                                truncation=self.config["truncation"],
                                inv_s=inv_s,
                                smoothness_std=smoothness_std,
                                use_view_dirs=self.config["decoder"]["radiance"]["use_view_dirs"],
                                use_normals=self.config["decoder"]["radiance"]["use_normals"],
                                concat_qp_to_rgb=self.config["decoder"]["radiance"]["concat_qp"],
                                concat_qp_to_occ=self.config["decoder"]["geometry"]["concat_qp"],
                                concat_dot_prod_to_rgb=self.config["decoder"]["radiance"]["use_dot_prod"],
                                rgb_feature_dim=self.config["rgb_feature_dim"],
                                semantic_feat_dim=self.config["semantic_feature_dim"],
                                iter=iter)
        
        rendered_rgb, rendered_depth = rend_dict["rgb"], rend_dict["depth"]
        rgb_loss = compute_loss(rendered_rgb, target_rgb_select, "l2")
        psnr = mse2psnr(rgb_loss)        
        valid_depth_mask = target_depth_select > 0.
        depth_loss = F.l1_loss(target_depth_select[valid_depth_mask], rendered_depth[valid_depth_mask])
        
        occ = rend_dict["occs"].clamp(1e-6, 1-1e-6)
        entropy_loss = -(occ*torch.log(occ) + (1-occ)*torch.log(1-occ)).mean()

        semantic_loss = (1 - torch.nn.CosineSimilarity()(rend_dict["semantic"], feat_2d)).mean()    
        if self.config["dino"]:
            dino_loss = (1 - torch.nn.CosineSimilarity()(rend_dict["dino"], dino_feat_2d)).mean()
        else:
            dino_loss = torch.zeros([1]).to(feat_2d.device)
        
        ret = {
            "rgb_loss": rgb_loss,
            "depth_loss": depth_loss,
            "semantic_loss":semantic_loss,
            "dino_loss": dino_loss,
            "occ_loss": rend_dict["occ_loss"],
            "fs_loss": rend_dict["fs_loss"],
            "entropy_loss": entropy_loss,
            "psnr": psnr
        }
        return ret
    
    def evaluate(self, rays_o, rays_d, target_rgb_select, target_depth_select):
        rend_dict = render_rays(self.occ_decoder,
                                self.rgb_decoder,
                                self.semantic_decoder,
                                self.dino_decoder,
                                self.grid,
                                self.semantic_grid,
                                self.volume_origin,
                                self.world_dims,
                                self.config["voxel_sizes"],
                                rays_o,
                                rays_d,
                                near=self.config["near"],
                                far=self.config["far"],
                                n_samples=self.config["n_samples"],
                                depth_gt=target_depth_select,
                                n_importance=self.config["n_importance"],
                                truncation=self.config["truncation"],
                                use_view_dirs=self.config["decoder"]["radiance"]["use_view_dirs"],
                                use_normals=self.config["decoder"]["radiance"]["use_normals"],
                                concat_qp_to_rgb=self.config["decoder"]["radiance"]["concat_qp"],
                                concat_qp_to_occ=self.config["decoder"]["geometry"]["concat_qp"],
                                concat_dot_prod_to_rgb=self.config["decoder"]["radiance"]["use_dot_prod"],
                                rgb_feature_dim=self.config["rgb_feature_dim"],
                                semantic_feat_dim=self.config["semantic_feature_dim"])

        rendered_rgb, rendered_depth = rend_dict["rgb"], rend_dict["depth"]
        rendered_semantic = rend_dict["semantic"]
        return rendered_rgb, rendered_depth, rendered_semantic

def batchify(fn, max_chunk=1024*128):
    if max_chunk is None:
        return fn
    def ret(feats):
        chunk = max_chunk // (feats.shape[1] * feats.shape[2])
        return torch.cat([fn(feats[i:i+chunk]) for i in range(0, feats.shape[0], chunk)], dim=0)
    return ret

def render_rays(occ_decoder,
                rgb_decoder,
                semantic_decoder,
                dino_decoder,
                feat_volume, 
                semantic_volume,
                volume_origin,
                volume_dim,
                voxel_sizes,
                rays_o,
                rays_d,
                truncation=0.10,
                near=0.01,
                far=3.0,
                n_samples=128,
                n_importance=16,
                depth_gt=None,
                inv_s=20.,
                normals_gt=None,
                smoothness_std=0.0,
                randomize_samples=True,
                use_view_dirs=False,
                use_normals=False,
                concat_qp_to_rgb=False,
                concat_qp_to_occ=False,
                concat_dot_prod_to_rgb=False,
                iter=0,
                rgb_feature_dim=[],
                semantic_feat_dim=[]
                ):
    n_rays = rays_o.shape[0]
    z_vals = torch.linspace(near, far, n_samples).to(rays_o)
    z_vals = z_vals[None, :].repeat(n_rays, 1)
    
    if depth_gt is not None:
        z_samples = torch.linspace(-0.25, 0.25, steps=n_importance).to(rays_o)
        z_samples = z_samples[None, :].repeat(n_rays, 1) + depth_gt[:,None]
        z_samples[depth_gt<=0] = torch.linspace(near, far, steps=n_importance).to(rays_o)

    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
    if randomize_samples:
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        z_vals = lower + (upper - lower) * torch.rand(z_vals.shape).to(rays_o)

    view_dirs = F.normalize(rays_d, dim=-1)[:, None, :].repeat(1, n_samples + n_importance, 1)
    query_points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
    query_points = query_points.requires_grad_(True)
    occ, rgb_feat, world_bound_mask, semantic_feat = qp_to_occ(query_points, volume_origin, volume_dim, feat_volume, semantic_volume,
                                                occ_decoder, concat_qp=concat_qp_to_occ, rgb_feature_dim=rgb_feature_dim, semantic_feat_dim = semantic_feat_dim)
    grads = None
    rgb_feat = [rgb_feat]
    
    if use_view_dirs:
        rgb_feat.append(view_dirs)
    if use_normals:
        rgb_feat.append(grads)
    if concat_qp_to_rgb:
        rgb_feat.append(2. * (query_points - volume_origin) / volume_dim - 1.)
    if concat_dot_prod_to_rgb:
        rgb_feat.append((view_dirs * grads).sum(dim=-1, keepdim=True))
    rgb = torch.sigmoid(rgb_decoder(torch.cat(rgb_feat, dim=-1)))

    semantic = semantic_decoder(semantic_feat)
    dino_pred = dino_decoder(semantic_feat)
    weights = occ2weight(occ)
    weights[~world_bound_mask] = 0.
    
    rendered_rgb = torch.sum(weights[..., None] * rgb, dim=-2)
    rendered_depth = torch.sum(weights * z_vals, dim=-1)
    rendered_semantic = torch.sum(weights[..., None] * semantic, dim=-2)

    rendered_semantic = rendered_semantic / (torch.linalg.norm(rendered_semantic, dim=-1, keepdim=True) + 1e-5)
    rendered_dino = torch.sum(weights[..., None] * dino_pred, dim=-2)
    fs_loss, occ_loss = get_occ_loss(occ, depth_gt[:, None], z_vals, truncation)
    
    ret = {"rgb": rendered_rgb,
           "depth": rendered_depth,
           "occ_loss": occ_loss,
           "fs_loss": fs_loss,
           "occs": occ,
           "weights": weights,
           "semantic": rendered_semantic,
           "dino": rendered_dino,
           }
    return ret

def occ2weight(occ):
    weights = occ * torch.cumprod(torch.cat([torch.ones([occ.shape[0], 1], device=occ.device), 1. - occ + 1e-7], -1), -1)[:, :-1]
    return weights

def qp_to_occ(pts, volume_origin, volume_dim, feat_volume, semantic_volume, occ_decoder, occ_act=nn.Identity(), concat_qp=False, rgb_feature_dim=[], semantic_feat_dim = []):
    pts_norm = 2. * (pts - volume_origin[None, None, :]) / volume_dim[None, None, :] - 1.  
    mask = (pts_norm.abs() <= 1.).all(dim=-1)
    pts_norm = pts_norm[mask].unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    mlvl_feats = feat_volume(pts_norm[...,[2,1,0]], concat=False)
    occ_feats = list(map(lambda feat_pts, rgb_dim: feat_pts[:,:-rgb_dim,...] if rgb_dim > 0 else feat_pts,
                         mlvl_feats, rgb_feature_dim))
    occ_feats = torch.cat(occ_feats, dim=1).squeeze(0).squeeze(1).squeeze(1).t()
    
    rgb_feats = map(lambda feat_pts, rgb_dim: feat_pts[:,-rgb_dim:,...] if rgb_dim > 0 else None,
                    mlvl_feats, rgb_feature_dim)
    rgb_feats = list(filter(lambda x: x is not None, rgb_feats))
    rgb_feats = torch.cat(rgb_feats, dim=1).squeeze(0).squeeze(1).squeeze(1).t()

    semantic_feats = semantic_volume(pts_norm[...,[2,1,0]], concat=False)
    semantic_feats = torch.cat(semantic_feats, dim=1).squeeze(0).squeeze(1).squeeze(1).t()
    
    rgb_feats_unmasked = torch.zeros(list(mask.shape) + [sum(rgb_feature_dim)], device=pts_norm.device)
    rgb_feats_unmasked[mask] = rgb_feats
    
    if concat_qp:
        occ_feats.append(pts_norm.permute(0,4,1,2,3))
    
    raw = occ_decoder(occ_feats)
    occ = torch.zeros_like(mask, dtype=pts_norm.dtype)
    occ[mask] = occ_act(raw.squeeze(-1))

    semantic_feats_unmasked = torch.zeros(list(mask.shape) + [sum(semantic_feat_dim)], device=pts_norm.device)
    semantic_feats_unmasked[mask] = semantic_feats
    
    return occ, rgb_feats_unmasked, mask, semantic_feats_unmasked

def neus_weights(occ, dists, inv_s, cos_val, z_vals=None):    
    estimated_next_occ = occ + cos_val * dists * 0.5
    estimated_prev_occ = occ - cos_val * dists * 0.5
    
    prev_cdf = torch.sigmoid(estimated_prev_occ * inv_s)
    next_cdf = torch.sigmoid(estimated_next_occ * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
    weights = alpha * torch.cumprod(torch.cat([torch.ones([occ.shape[0], 1], device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    
    if z_vals is not None:
        signs = occ[:, 1:] * occ[:, :-1]
        mask = torch.where(signs < 0., torch.ones_like(signs), torch.zeros_like(signs))
        inds = torch.argmax(mask, dim=1, keepdim=True)
        z_surf = torch.gather(z_vals, 1, inds)
        return weights, z_surf
    
    return weights

def apply_log_transform(tocc):
    sgn = torch.sign(tocc)
    out = torch.log(torch.abs(tocc) + 1)
    out = sgn * out
    return out

def compute_loss(prediction, target, loss_type="l2"):
    if loss_type == "l2":
        return F.mse_loss(prediction, target)
    elif loss_type == "l1":
        return F.l1_loss(prediction, target)
    elif loss_type == "log":
        return F.l1_loss(apply_log_transform(prediction), apply_log_transform(target))
    raise Exception("Unknown loss type")

def compute_grads(predicted_occ, query_points):
    occ_grad, = torch.autograd.grad([predicted_occ], [query_points], [torch.ones_like(predicted_occ)], create_graph=True)
    return occ_grad

def get_occ_loss(predicted_occ, target_d, z_vals, truncation):
    depth_mask = target_d > 0.
    front_mask = (z_vals < (target_d - truncation))
    front_mask = (front_mask | ((target_d < 0.) & (z_vals < 3.5)))
    occ_mask = ((target_d - z_vals).abs() <= truncation) & depth_mask

    predicted_occ = predicted_occ.view(-1,1)
    gt_front = torch.cat([torch.zeros_like(predicted_occ), torch.ones_like(predicted_occ)], dim=-1)
    gt_middle = torch.cat([torch.ones_like(predicted_occ), torch.zeros_like(predicted_occ)], dim=-1)
    predicted_occ = torch.cat([predicted_occ, 1-predicted_occ], dim=-1)
    front_mask = front_mask.view(-1,1)
    occ_mask = occ_mask.view(-1,1)

    fs_loss = BCELoss(predicted_occ*front_mask, gt_front * front_mask)
    occ_loss = BCELoss(predicted_occ*occ_mask, gt_middle * occ_mask)

    return fs_loss, occ_loss

def get_occ_loss(z_vals, target_d, predicted_occ, truncation):
    depth_mask = target_d > 0.
    front_mask = (z_vals < (target_d - truncation))
    front_mask = (front_mask | ((target_d < 0.) & (z_vals < 3.5)))
    bound = (target_d - z_vals) 
    bound[target_d[:,0] < 0., :] = 10.
    occ_mask = (bound.abs() <= truncation) & depth_mask
    
    sum_of_samples = front_mask.sum(dim=-1) + occ_mask.sum(dim=-1) + 1e-8
    rays_w_depth = torch.count_nonzero(target_d)
    
    fs_loss = (torch.max(torch.exp(-5. * predicted_occ) - 1., predicted_occ - bound).clamp(min=0.) * front_mask)
    fs_loss = (fs_loss.sum(dim=-1) / sum_of_samples).sum() / rays_w_depth
    occ_loss = ((torch.abs(predicted_occ - bound) * occ_mask).sum(dim=-1) / sum_of_samples).sum() / rays_w_depth

    return fs_loss, occ_loss

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    device = weights.device
    weights = weights + 1e-5 
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1], device=device), cdf], -1)
    if det:
        u = torch.linspace(0. + 0.5 / N_importance, 1. - 0.5 / N_importance, steps=N_importance, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=device), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples