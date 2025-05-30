import numpy as np

def get_scene_bounds(scene_name):
    if scene_name == "2t7WUuJeko7":
        x_min, x_max = -1.6, 10.2
        y_min, y_max = -7.5, 2.0
        z_min, z_max = -0.2, 3.5
    
    elif scene_name == "jh4fc5c5qoQ":
        x_min, x_max = -2.5, 14.8
        y_min, y_max = -5.8, 9.0
        z_min, z_max = -0.2, 7.2

    elif scene_name == "zsNo4HB9uLZ":
        x_min, x_max = -3.0, 19.3
        y_min, y_max = -5.3, 8.2
        z_min, z_max = -0.1, 2.7

    elif scene_name == 'scene0000_00':
        x_min, x_max = -0.2, 8.6
        y_min, y_max = -0.2, 8.9
        z_min, z_max = -0.2, 3.2

    elif scene_name == "office0":
        x_min, x_max = -3, 3
        y_min, y_max = -4, 2.5
        z_min, z_max = -2, 2.5

    elif scene_name == "office1":
        x_min, x_max = -2, 3.2
        y_min, y_max = -1.7, 2.7
        z_min, z_max = -1.2, 2.0
    
    elif scene_name == "office2":
        x_min, x_max = -3.6, 3.2
        y_min, y_max = -3.0, 5.5
        z_min, z_max = -1.4, 1.7
    
    elif scene_name == "office3":
        x_min, x_max = -5.3, 3.7
        y_min, y_max = -6.1, 3.4
        z_min, z_max = -1.4, 2.0

    elif scene_name == "office4":
        x_min, x_max = -1.4, 5.5
        y_min, y_max = -2.5, 4.4
        z_min, z_max = -1.4, 1.8

    elif scene_name == "room0":
        x_min, x_max = -1.0, 7.0
        y_min, y_max = -1.3, 3.7
        z_min, z_max = -1.7, 1.4

    elif scene_name == "room1":
        x_min, x_max = -5.6, 1.4
        y_min, y_max = -3.2, 2.8
        z_min, z_max = -1.6, 1.8

    elif scene_name == "room2":
        x_min, x_max = -1.0, 6.1
        y_min, y_max = -3.4, 1.9
        z_min, z_max = -3.1, 0.8
    else:
        raise NotImplementedError

    return np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])

