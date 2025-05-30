import glob, os
import numpy as np
import plyfile
import torch
import pandas as pd
from plyfile import *

remapper = np.ones(150) * (27)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 10, 13, 18, 11, 22, 24, 25, 32, 33, 34, 36, 35, 31, 21, 23, 19]):
    remapper[x] = i
remapper[15] = 11
remapper[16] = 12

MATTERPORT_COLOR_MAP_label35_ours = {
0 :( 174 , 199 , 232 ) ,
1 :( 152 , 223 , 138 ) ,
2 :( 31 , 119 , 180 ) ,
3 :( 255 , 187 , 120 ) ,
4 :( 188 , 189 , 34 ) ,
5 :( 140 , 86 , 75 ) ,
6 :( 255 , 152 , 150 ) ,
7 :( 214 , 39 , 40 ) ,
8 :( 197 , 176 , 213 ) ,
9 :( 23 , 190 , 207 ) ,
10 :( 247 , 182 , 210 ) ,
11 :( 66 , 188 , 102 ) ,
12 :( 219 , 219 , 141 ) ,
13 :( 202 , 185 , 52 ) ,
14 :( 51 , 176 , 203 ) ,
15 :( 78 , 71 , 183 ) ,
16 :( 255 , 127 , 14 ) ,
17 :( 91 , 163 , 138 ) ,
18 :( 146 , 111 , 194 ) ,
19 :( 44 , 160 , 44 ) ,
20 :( 112 , 128 , 144 ) ,
21 :( 153.0 , 108.0 , 234.0 ) ,
22: (237.0, 204.0, 37.0), 
23: (220, 20, 60), 
24: (34.0, 14.0, 130.0), 
25: (192.0, 229.0, 91.0), 
26: (162.0, 62.0, 60.0), 
27 :( 118.0 , 174.0 , 76.0 ) , 
28 :( 82 , 84 , 163 ) , 
29: (150.0, 53.0, 56.0), 
30: (64.0, 158.0, 70.0), 
31: (208.0, 49.0, 84.0), 
32 :( 143.0 , 45.0 , 115.0 ) , 
33: ( 102.0 , 255.0 , 255.0 ) , 
34 :( 0 , 0 , 0 ) ,
}

base_path = 'your path to dataset'
matterport_path = os.path.join(base_path, 'Matterport3D/v1/scans')
tsv_file = os.path.join(base_path, 'Matterport3D/v1/scans/category_mapping.tsv')
out_dir = 'script/matterport_3d/label35'
os.makedirs(out_dir, exist_ok=True)
scene_names = ["2t7WUuJeko7"]

category_mapping = pd.read_csv(tsv_file, sep='\t', header=0)
mapping = np.insert(category_mapping[['nyu40id']].to_numpy()
                        .astype(int).flatten(), 0, 0, axis=0)

nyu40id = category_mapping[['nyu40id']].to_numpy().flatten().astype(np.int32)
nyuClass = category_mapping[['nyuClass']].to_numpy().flatten()
nyu40id[np.isnan(nyu40id)] = 0
N_max = np.max(nyu40id)

files = []
for scene_name in scene_names:
    files = files + glob.glob(os.path.join(matterport_path, scene_name, 'region_segmentations', '*.ply'))

for fn in files:
    scene_name = fn.split('/')[-3]
    region_name = fn.split('/')[-1].split('.')[0]
    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    faces_in = a.elements[1]
    coords = np.ascontiguousarray(v[:, :3])
    colors = np.ascontiguousarray(v[:, -3:])

    category_id = a['face']['category_id']
    category_id[category_id==-1] = 0
    mapped_labels = mapping[category_id]
    remapped_labels = remapper[mapped_labels].astype(int)

    triangles = a['face']['vertex_indices']
    vertex_labels = np.zeros((coords.shape[0], 28), dtype=np.int32)
    new_colors = np.zeros((coords.shape[0], 3), dtype=np.int32)
    for row_id in range(triangles.shape[0]):
        for i in range(3):
            vertex_labels[triangles[row_id][i],
                            remapped_labels[row_id]] += 1

    vertex_labels = np.argmax(vertex_labels, axis=1)
    vertex_labels[vertex_labels==27] = 34

    for i in range(vertex_labels.shape[0]):
        new_colors[i] = np.array(MATTERPORT_COLOR_MAP_label35_ours[vertex_labels[i]])
    for i in range(v.shape[0]):
        v[i, -3:] = new_colors[i]

    vertices_new = []
    for i in range(v.shape[0]):
        vertices_new.append((v[i,0], v[i,1], v[i,2], v[i,3], v[i,4], v[i,5], v[i,6], v[i,7], v[i,8], v[i,9], v[i,10]))
    path_out = os.path.join(out_dir, scene_name + "{}_semantic.ply".format(region_name))
    vertices_new = PlyElement.describe(np.array(vertices_new, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('tx', 'f4'), ('ty', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
    # PlyData([vertices_new, faces_in]).write(path_out)

    labels = np.array(vertex_labels, dtype=np.int32)
    labels[labels == 34] = 255
    torch.save((coords, colors, labels), os.path.join(out_dir, scene_name + '{}.pth'.format(region_name)))

