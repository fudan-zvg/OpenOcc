MATTERPORT_LABELS_label35 = ['wall', 'floor', 'cabinet', 'bed', 'chair', 
                        'sofa', 'table', 'door', 'window', 'counter',
                        'desk', 'shelf', 'curtain', 'pillow', 'picture', 
                        'ceiling', 'refrigerator', 'tv', 'nightstand', 'toilet', 
                        'sink',  'bathtub',  'lamp', 'person', 'clothes', 
                        'book',  'mirror', 'backpack', 'trash can', 'plant',
                        'banister', 'stairs',  'stool', 'vase', "other"]
label_id = [ 0,  1,  2,  3,  4,  6,  7,  8,  9, 12, 14, 15, 16, 18, 19, 22 ,24, 25 ,32, 33, 34, 38, 39, 40]
MATTERPORT_COLOR_MAP_label35 = {
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

my_colors = []
for _, value in MATTERPORT_COLOR_MAP_label35.items():
    my_colors.append(list(value))

import matplotlib.pyplot as plt
names = []
colors = []
labels = []
colleges = []
for i, color in enumerate(my_colors):
    names.append(str(i))
    color = [color[0] / 255.0, color[1] / 255.0, color[2] / 255.0 ]
    colors.append(color)
    labels.append(MATTERPORT_LABELS_label35[i])
    colleges.append(75 - i *2)

fig,ax=plt.subplots()
b=ax.barh(range(len(names)),colleges,color=colors)

for i, rect in enumerate(b):
    w=rect.get_width()
    ax.text(w,rect.get_y()+rect.get_height()/2,labels[i],ha='left',va='center')

ax.get_xaxis().set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig("script/matterport35_label.png")