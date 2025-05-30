import os
import numpy as np
import cv2
import imageio
import torch

from model.utils import normalize_rgb
from dataio.dino_extractor import ViTExtractor
from dataio.get_scene_bounds import get_scene_bounds
from dataio.openseg_extractor import extract_openseg_img_feature

import tensorflow as tf2
import tensorflow.compat.v1 as tf

class ScannetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_dir: str,
                 device,
                 start: int = -1,
                 end: int = -1,
                 trainskip: int = 1,
                 near: float = 0.01,
                 far: float = 5.0,
                 norm_rgb=False,
                 load=True,
                 openseg = True,
                 dino = False
                 ):
        super(ScannetDataset).__init__()

        self.device = device
        self.dataset_dir = dataset_dir

        self.rgb_dir = os.path.join(self.dataset_dir, "image/color")
        self.depth_dir = os.path.join(self.dataset_dir, "image/depth")
        self.pose_dir = os.path.join(self.dataset_dir, "image/pose")
        self.rgb_pattern = "{:d}.jpg"
        self.depth_pattern = "{:d}.png"
        self.pose_pattern = "{:d}.txt"
        intri_rgb = np.loadtxt(os.path.join(dataset_dir, "image/intrinsic/intrinsic_color.txt")).astype(np.float32)[:3, :3]
        intri_depth = np.loadtxt(os.path.join(dataset_dir, "image/intrinsic/intrinsic_depth.txt")).astype(np.float32)[:3, :3]
        self.intrinsics_rgb = torch.from_numpy(intri_rgb)
        self.intrinsics_depth = torch.from_numpy(intri_depth)

        n_frames = len(os.listdir(self.pose_dir))
        self.scene_name = dataset_dir.split("/")[-1]
        self.H = 480
        self.W = 640
        self.intri_depth = intri_depth
        self.near = near
        self.far = far
        self.norm_rgb = norm_rgb
        self.frame_ids = []
        self.load = load

        if self.load:
            self.depth_cleaner = cv2.rgbd.DepthCleaner_create(cv2.CV_32F)
            self.normal_estimator = cv2.rgbd.RgbdNormals_create(self.H, self.W, cv2.CV_32F, intri_depth, 5)

        self.c2w_list = []
        self.rgb_list = []
        self.depth_list = []
        self.K_list = []

        self.feats_list = []
        self.openseg = openseg
        # Note: your path to openseg_exported_clip
        self.model_path = 'path/openseg_exported_clip'
        self.feat_dim = 768

        self.dino_model_type = "dino_vits8"
        self.dino_stride = 4
        self.dino_load_size = 600
        self.dino_layer = 11
        self.dino_facet = "key"
        self.dino_bin = False
        self.dino_list = []
        self.dino = dino

        base_dir = ("/").join(dataset_dir.split("/")[:-1])
        feat_2d_path = os.path.join(base_dir, "clip")
        file = os.path.join(feat_2d_path, 'scannet_{}_{}.pt'.format(self.scene_name, trainskip))
        self.openseg_already = False
        if os.path.exists(file):
            self.openseg_already = True
            if self.openseg:
                self.feats_list = torch.load(file)

        dino_2d_path = os.path.join(base_dir, "dino")
        dino_file = os.path.join(dino_2d_path, 'scannet_{}_{}.pt'.format(self.scene_name,trainskip))
        self.dino_already = False
        if os.path.exists(dino_file):
            self.dino_already = True
            if self.dino:
                self.dino_list = torch.load(dino_file)

        for i in range(n_frames):
            if i < start:
                continue
            if i == end:
                break
            if (i - start) % trainskip != 0:
                continue

            c2w = np.loadtxt(os.path.join(self.pose_dir, self.pose_pattern.format(i))).astype(np.float32).reshape(4, 4)
            c2w = torch.from_numpy(c2w)
            if torch.isnan(c2w).any() or torch.isinf(c2w).any():
                continue

            self.frame_ids.append(i)
            self.c2w_list.append(c2w)
        
        if self.openseg and not self.openseg_already:
            self.openseg_model = tf2.saved_model.load(self.model_path,
                    tags=[tf.saved_model.tag_constants.SERVING],)
            self.text_emb = tf.zeros([1, 1, self.feat_dim])

        if self.dino and not self.dino_already:
            self.dino_extractor = ViTExtractor(self.dino_model_type, self.dino_stride)

        if load:
            self.get_all_frames()

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["depth_cleaner"]
        del d["normal_estimator"]
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.depth_cleaner = cv2.rgbd.DepthCleaner_create(cv2.CV_32F, 5) 
        self.normal_estimator = cv2.rgbd.RgbdNormals_create(self.H, self.W, cv2.CV_32F, self.intri_depth, 5)

    def get_bounds(self):
        return torch.from_numpy(get_scene_bounds(self.dataset_dir.split('/')[-1])).float()

    def get_all_frames(self):
        for i, frame_id in enumerate(self.frame_ids):
            rgb_path = os.path.join(self.rgb_dir, self.rgb_pattern.format(frame_id))
            depth_path = os.path.join(self.depth_dir, self.depth_pattern.format(frame_id))

            rgb = np.array(imageio.imread(rgb_path)).astype(np.float32)
            h_orig = rgb.shape[0]
            w_orig = rgb.shape[1]
            rgb = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_AREA)
            rgb = torch.as_tensor(rgb).permute(2, 0, 1)
            s_h = float(h_orig) / float(self.H)
            s_w = float(w_orig) / float(self.W)
            intri_rgb = self.intrinsics_rgb.clone()
            intri_rgb[0, :] /= s_w
            intri_rgb[1, :] /= s_h

            if self.norm_rgb:
                rgb = normalize_rgb(rgb)
            else:
                rgb /= 255.

            if self.openseg and not self.openseg_already:
                feat_2d = extract_openseg_img_feature(rgb_path, self.openseg_model, self.text_emb, img_size=[480, 640])
            
            if self.dino and not self.dino_already:
                dino_feat_2d = self.extract_dino_img_feature(rgb_path, img_size=[480, 640])

            depth = cv2.imread(depth_path, -1).astype(np.float32)
            depth_filtered = self.depth_cleaner.apply(depth) / 1000.

            depth_filtered[depth == 0.] = 0.
            depth_filtered = np.nan_to_num(depth_filtered)
            depth = torch.from_numpy(depth_filtered)
            depth[depth < self.near] = 0.
            depth[depth > self.far] = 0.

            self.rgb_list.append(rgb)
            self.depth_list.append(depth)
            self.K_list.append(self.intrinsics_depth)

            if self.openseg and not self.openseg_already:
                self.feats_list.append(feat_2d)

            if self.dino and not self.dino_already:
                dino_feat_upsample = torch.nn.functional.interpolate(dino_feat_2d[None, ...], size=[480, 640], mode='bilinear',
                                                align_corners=False)
                dino_feat_2d = dino_feat_upsample.squeeze(0)
                self.dino_list.append(dino_feat_2d)

        self.rgb_list = torch.stack(self.rgb_list, dim=0)
        self.depth_list = torch.stack(self.depth_list, dim=0)
        self.K_list = torch.stack(self.K_list, dim=0)

        if self.openseg and not self.openseg_already:
            self.feats_list = torch.stack(self.feats_list, dim = 0)
        if self.dino and not self.dino_already:
            self.dino_list = torch.stack(self.dino_list, dim = 0)

    def get_frame(self, id):
        ret = {
            "frame_id": self.frame_ids[id],
            "sample_id": id,
            "c2w": self.c2w_list[id],
            "rgb": self.rgb_list[id],
            "depth": self.depth_list[id],
            "K": self.intrinsics_depth,
            "feat_2d": self.feats_list[id] if self.openseg else None,
            "dino_2d": self.dino_list[id] if self.dino else None
        }
        return ret

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, id):
        return self.get_frame(id)
    
    def extract_dino_img_feature(self, image_dir, img_size=None):
        preproc_image = self.dino_extractor.preprocess(image_dir, img_size).to(torch.device("cuda"))
        with torch.no_grad():
            descriptors = self.dino_extractor.extract_descriptors(
                preproc_image,
                [self.dino_layer],
                self.dino_facet,
                self.dino_bin,
            )
        descriptors = descriptors.reshape(self.dino_extractor.num_patches[0], self.dino_extractor.num_patches[1], -1)
        descriptors = descriptors.cpu().detach().permute(2, 0 ,1)
        return descriptors

if __name__ == '__main__':
    scene_names = ["scene0000_00"] 
    trainskip = 50

    dataset_dir = "your data path"
    feat_2d_path = os.path.join(dataset_dir, "clip")
    dino_feat_2d_path = os.path.join(dataset_dir, "dino")
    os.makedirs(feat_2d_path, exist_ok=True)
    os.makedirs(dino_feat_2d_path, exist_ok=True)

    for scene_name in scene_names:
        print(scene_name)
        data_path = os.path.join(dataset_dir, scene_name)
        dataset = ScannetDataset(data_path, trainskip=trainskip, openseg=True, dino=False, device=torch.device("cpu"))
        feat_2d = dataset.feats_list
        dino_2d = dataset.dino_list

        if not dataset.openseg_already:
            torch.save(feat_2d,  os.path.join(feat_2d_path, 'scannet_{}_{}.pt'.format(scene_name, trainskip)))
        if not dataset.dino_already:
            torch.save(dino_2d, os.path.join(dino_feat_2d_path, 'scannet_{}_{}.pt'.format(scene_name, trainskip)))
