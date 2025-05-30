import os
import numpy as np
import cv2
import imageio
import glob
import torch

from model.utils import normalize_rgb
from dataio.dino_extractor import ViTExtractor
from dataio.get_scene_bounds import get_scene_bounds
from dataio.openseg_extractor import extract_openseg_img_feature

import tensorflow as tf2
import tensorflow.compat.v1 as tf

class MatterportDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_dir: str,
                 device,
                 trainskip: int = 1,
                 near: float = 0.01,
                 far: float = 8.0,
                 norm_rgb=False,
                 load=True,
                 openseg = True,
                 dino = False,
                 i_start = None, 
                 i_end = None
                 ):
        super(MatterportDataset).__init__()

        self.device = device
        self.dataset_dir = dataset_dir

        self.img_files = sorted(glob.glob(f'{self.dataset_dir}/color/*.jpg'))
        if i_start is not None and i_end is not None:
            self.img_files = self.img_files[i_start:i_end]
        self.scene_name = dataset_dir.split("/")[-1]

        self.extrinsics = []
        self.intrinsics = []
        for img_name in self.img_files:
            name = img_name.split('/')[-1][:-4]
            self.extrinsics.append(np.loadtxt(os.path.join(os.path.join(self.dataset_dir, 'pose'), name+'.txt')))
            self.intrinsics.append(np.loadtxt(os.path.join(os.path.join(self.dataset_dir, 'intrinsic'), name+'.txt')))
        self.intrinsics = np.stack(self.intrinsics, axis=0)
        self.intrinsics = torch.from_numpy(self.intrinsics).float()
        self.extrinsics = np.stack(self.extrinsics, axis=0)
        self.extrinsics = torch.from_numpy(self.extrinsics).float()

        self.H = 512
        self.W = 640
        self.near = near
        self.far = far
        self.norm_rgb = norm_rgb
        self.frame_ids = []
        self.load = load

        self.c2w_list = []
        self.rgb_list = []
        self.depth_list = []
        self.K_list = []
        self.trainskip = trainskip

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

        N_img= len(self.img_files)
        N_setp = 100
        base_dir = ("/").join(dataset_dir.split("/")[:-1])
        self.feat_2d_path = os.path.join(base_dir, "clip")
        file = os.path.join(self.feat_2d_path, 'matterport_{}_{}_{}.pt'.format(self.scene_name, 0, 100))
        self.openseg_already = False
        if os.path.exists(file):
            self.openseg_already = True
            if self.openseg:
                for i_start in range(0, N_img, N_setp):
                    i_end = (i_start + N_setp) if (i_start + N_setp) < N_img else N_img
                    file = os.path.join(self.feat_2d_path, 'matterport_{}_{}_{}.pt'.format(self.scene_name, i_start, i_end))
                    self.feats_list.append(torch.load(file))
        self.feats_list_len = len(self.feats_list)
        
        dino_2d_path = os.path.join(base_dir, "dino")
        dino_file = os.path.join(dino_2d_path, 'matterport_{}_{}_{}.pt'.format(self.scene_name, 0, 100))
        self.dino_already = False
        if os.path.exists(dino_file):
            self.dino_already = True
            if self.dino:
                for i_start in range(0, N_img, N_setp):
                    i_end = (i_start + N_setp) if (i_start + N_setp) < N_img else N_img
                    dino_file = os.path.join(dino_2d_path, 'matterport_{}_{}_{}.pt'.format(self.scene_name, i_start, i_end))
                    self.dino_list.append(torch.load(dino_file))
                self.dino_list = torch.cat(self.dino_list, dim = 0)

        if self.openseg and not self.openseg_already:
            self.openseg_model = tf2.saved_model.load(self.model_path,
                    tags=[tf.saved_model.tag_constants.SERVING],)
            self.text_emb = tf.zeros([1, 1, self.feat_dim])
        
        if self.dino and not self.dino_already:
            self.dino_extractor = ViTExtractor(self.dino_model_type, self.dino_stride)

        if load:
            self.get_all_frames()
        
        self.n_frames = len(self.c2w_list)
        
    def get_bounds(self):
        return torch.from_numpy(get_scene_bounds(self.dataset_dir.split('/')[-1])).float()

    def get_all_frames(self):
        for frame_id, img_path in enumerate(self.img_files):
            if frame_id % self.trainskip != 0:
                continue
            self.frame_ids.append(frame_id)

            pose = self.extrinsics[frame_id]
            self.c2w_list.append(pose)

            rgb = np.array(imageio.imread(img_path)).astype(np.float32)
            rgb = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_AREA)
            rgb = torch.as_tensor(rgb).permute(2, 0, 1)

            if self.norm_rgb:
                rgb = normalize_rgb(rgb)
            else:
                rgb /= 255.

            if self.openseg and not self.openseg_already:
                feat_2d = extract_openseg_img_feature(img_path, self.openseg_model, self.text_emb, img_size=[self.H, self.W])
            if self.dino and not self.dino_already:
                dino_feat_2d = self.extract_dino_img_feature(img_path, img_size=[self.H, self.W])

            depth_path = img_path.replace('color', 'depth')
            _, img_type, yaw_id = img_path.split('/')[-1].split('_')
            depth_path = depth_path[:-8] + 'd'+img_type[1] + '_' + yaw_id[0] + '.png'

            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth = depth / 4000.0
            depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            depth = torch.from_numpy(depth)
            depth[depth < self.near] = 0.
            depth[depth > self.far] = 0.

            self.rgb_list.append(rgb)
            self.depth_list.append(depth)
            K = self.intrinsics[frame_id]
            self.K_list.append(K)

            if self.openseg and not self.openseg_already:
                self.feats_list.append(feat_2d)
            
            if self.dino and not self.dino_already:
                dino_feat_upsample = torch.nn.functional.interpolate(dino_feat_2d[None, ...], size=[self.H, self.W], mode='bilinear',
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
            "K": self.K_list,
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
    scene_names = ["2t7WUuJeko7", "jh4fc5c5qoQ", "zsNo4HB9uLZ"]
    trainskip = 1
    dataset_dir = "your data path"
    feat_2d_save_path = os.path.join(dataset_dir, "clip")
    os.makedirs(feat_2d_save_path, exist_ok=True)

    for scene_name in scene_names:
        print(scene_name)
        data_path = os.path.join(dataset_dir, scene_name)
        img_files = sorted(glob.glob(f'{data_path}/color/*.jpg'))
        N_img= len(img_files)
        N_setp = 100

        for i_start in range(0, N_img, N_setp):
            i_end = (i_start + N_setp) if (i_start + N_setp) < N_img else N_img
            dataset = MatterportDataset(data_path, trainskip=trainskip, openseg=True, dino=False, device=torch.device("cpu"), i_start = i_start, i_end = i_end)
            feat_2d = dataset.feats_list

            if not dataset.openseg_already:
                torch.save(feat_2d,  os.path.join(feat_2d_save_path, 'matterport_{}_{}_{}.pt'.format(scene_name, i_start, i_end)))
        