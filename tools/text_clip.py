from tools.label_constants import *
import numpy as np
import clip

def precompute_text_related_properties(labelset_name):
    '''pre-compute text features, labelset, palette, and mapper.'''
    
    if "replica" in labelset_name:
        labelset = list(Replica_LABELS_label30)
        palette = get_palette(colormap='replica')

    elif "mp_label35" in labelset_name:
        labelset = list(MATTERPORT_LABELS_label35)
        palette = get_palette(colormap='mp_label35')

    else: # an arbitrary dataset, just use a large labelset
        labelset = list(MATTERPORT_LABELS_160)
        palette = get_palette(colormap='matterport_160')

    mapper = None
    text_features = extract_text_feature(labelset, labelset_name)
    return text_features, labelset, mapper, palette

def get_palette(num_cls=21, colormap='scannet'):
    if colormap == 'replica':
        label_palette = []
        for _, value in Replica_COLOR_MAP_label30.items():
            label_palette.append(np.array(value))
        palette = np.concatenate(label_palette)
    elif colormap == "mp_label35":
        label_palette = []
        for _, value in MATTERPORT_COLOR_MAP_label35.items():
            label_palette.append(np.array(value))
        palette = np.concatenate(label_palette)
    elif colormap == 'matterport_160':
        label_palette = []
        for _, value in MATTERPORT_COLOR_MAP_160.items():
            label_palette.append(np.array(value))
        palette = np.concatenate(label_palette)
    else:
        n = num_cls
        palette = [0]*(n*3)
        for j in range(0,n):
            lab = j
            palette[j*3+0] = 0
            palette[j*3+1] = 0
            palette[j*3+2] = 0
            i = 0
            while lab > 0:
                palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                i = i + 1
                lab >>= 3
    return palette

def extract_text_feature(labelset, labelset_name, feature_2d_extractor = "openseg"):
    '''extract CLIP text features.'''
    print('Use prompt engineering: a XX in a scene')
    labelset = [ "a " + label + " in a scene" for label in labelset]
    if 'replica' in labelset_name:
        labelset[-1] = 'other'
    if 'mp_label35' in labelset_name:
        labelset[-1] = 'other'

    if 'lseg' in feature_2d_extractor:
        text_features = extract_clip_feature(labelset)
    elif 'openseg' in feature_2d_extractor:
        text_features = extract_clip_feature(labelset, model_name="ViT-L/14@336px")
    else:
        raise NotImplementedError

    return text_features

def extract_clip_feature(labelset, model_name="ViT-B/32"):
    # "ViT-L/14@336px" # the big model that OpenSeg uses
    print("Loading CLIP {} model...".format(model_name))
    clip_pretrained, _ = clip.load(model_name, device='cuda', jit=False)
    print("Finish loading")

    if isinstance(labelset, str):
        lines = labelset.split(',')
    elif isinstance(labelset, list):
        lines = labelset
    else:
        raise NotImplementedError

    labels = []
    for line in lines:
        label = line
        labels.append(label)
    text = clip.tokenize(labels)
    text = text.cuda()
    text_features = clip_pretrained.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features

def convert_labels_with_palette(input, palette):
    '''Get image color palette for visualizing masks'''
    new_3d = np.zeros((input.shape[0], 3))
    u_index = np.unique(input)
    for index in u_index:
        if index == 255:
            index_ = 20
        else:
            index_ = index

        new_3d[input==index] = np.array(
            [palette[index_ * 3] / 255.0,
             palette[index_ * 3 + 1] / 255.0,
             palette[index_ * 3 + 2] / 255.0])
    return new_3d

def convert_labels_with_palette_vis(input, palette):
    '''Get image color palette for visualizing masks'''
    u_index = np.unique(input)
    new_3d_color_map = np.zeros((u_index.shape[0], 4))
    for index in u_index:
        if index == 255:
            index_ = 20
        else:
            index_ = index

        new_3d_color_map[index] = np.array(
            [palette[index_ * 3],
             palette[index_ * 3 + 1],
             palette[index_ * 3 + 2], 255])
    return new_3d_color_map