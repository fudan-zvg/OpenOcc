'''Label file for different datasets.'''

MATTERPORT_LABELS_160 = ('wall', 'door', 'ceiling', 'floor', 'picture', 'window', 'chair', 'pillow', 'lamp', 
                            'cabinet', 'curtain', 'table', 'plant', 'mirror', 'towel', 'sink', 'shelves', 'sofa', 
                            'bed', 'night stand', 'toilet', 'column', 'banister', 'stairs', 'stool', 'vase', 
                            'television', 'pot', 'desk', 'box', 'coffee table', 'counter', 'bench', 'garbage bin', 
                            'fireplace', 'clothes', 'bathtub', 'book', 'air vent', 'faucet', 'photo', 'toilet paper', 
                            'fan', 'railing', 'sculpture', 'dresser', 'rug', 'ottoman', 'bottle', 'refridgerator', 
                            'bookshelf', 'wardrobe', 'pipe', 'monitor', 'stand', 'drawer', 'container', 'light switch', 
                            'purse', 'door way', 'basket', 'chandelier', 'oven', 'clock', 'stove', 'washing machine', 
                            'shower curtain', 'fire alarm', 'bin', 'chest', 'microwave', 'blinds', 'bowl', 'tissue box', 
                            'plate', 'tv stand', 'shoe', 'heater', 'headboard', 'bucket', 'candle', 'flower pot', 
                            'speaker', 'furniture', 'sign', 'air conditioner', 'fire extinguisher', 'curtain rod', 
                            'floor mat', 'printer', 'telephone', 'blanket', 'handle', 'shower head', 'soap', 'keyboard', 
                            'thermostat', 'radiator', 'kitchen island', 'paper towel', 'sheet', 'glass', 'dishwasher', 
                            'cup', 'ladder', 'garage door', 'hat', 'exit sign', 'piano', 'board', 'rope', 'ball', 
                            'excercise equipment', 'hanger', 'candlestick', 'light', 'scale', 'bag', 'laptop', 'treadmill', 
                            'guitar', 'display case', 'toilet paper holder', 'bar', 'tray', 'urn', 'decorative plate', 'pool table', 
                            'jacket', 'bottle of soap', 'water cooler', 'utensil', 'tea pot', 'stuffed animal', 'paper towel dispenser', 
                            'lamp shade', 'car', 'toilet brush', 'doll', 'drum', 'whiteboard', 'range hood', 'candelabra', 'toy', 
                            'foot rest', 'soap dish', 'placemat', 'cleaner', 'computer', 'knob', 'paper', 'projector', 'coat hanger', 
                            'case', 'pan', 'luggage', 'trinket', 'chimney', 'person', 'alarm')

MATTERPORT_COLOR_MAP_160 = {
    1: (174., 199., 232.), # wall
    2: (214., 39., 40.), # door
    3: (186., 197., 62.), # ceiling
    4: (152., 223., 138.), # floor
    5: (196., 156., 148.), # picture
    6: (197., 176., 213.), # window
    7: (188., 189., 34.), # chair
    8: (141., 91., 229.), # pillow
    9: (237.0, 204.0, 37.0), # lamp
    10: (31., 119., 180.), # cabinet
    11: (219., 219., 141.), # curtain
    12: (255., 152., 150.), # table
    13: (150.0, 53.0, 56.0), # plant
    14: (162.0, 62.0, 60.0), # mirror
    15: (62.0, 143.0, 148.0), # towel
    16: (112., 128., 144.), # sink
    17: (229.0, 91.0, 104.0), # shelves
    18: (140., 86., 75.), # sofa
    19: (255., 187., 120.), # bed
    20: (137.0, 63.0, 14.0), # night stand
    21: (44., 160., 44.), # toilet
    22: (39.0, 19.0, 208.0), # column
    23: (64.0, 158.0, 70.0), # banister
    24: (208.0, 49.0, 84.0), # stairs
    25: (90.0, 119.0, 201.0), # stool
    26: (118., 174., 76.), # vase
    27: (143.0, 45.0, 115.0), # television
    28: (153., 108., 234.), # pot
    29: (247., 182., 210.), # desk
    30: (177.0, 82.0, 239.0), # box
    31: (58.0, 98.0, 137.0), # coffee table
    32: (23., 190., 207.), # counter
    33: (17.0, 242.0, 171.0), # bench
    34: (79.0, 55.0, 137.0), # garbage bin
    35: (127.0, 63.0, 52.0), # fireplace
    36: (34.0, 14.0, 130.0), # clothes
    37: (227., 119., 194.), # bathtub
    38: (192.0, 229.0, 91.0), # book
    39: (49.0, 206.0, 87.0), # air vent
    40: (250., 253., 26.), # faucet
    41: (0., 0., 0.), # unlabel/unknown
    80: (82., 75., 227.),
    82: (253., 59., 222.),
    84: (240., 130., 89.),
    86: (123., 172., 47.),
    87: (71., 194., 133.),
    88: (24., 94., 205.),
    89: (134., 16., 179.),
    90: (159., 32., 52.),
    93: (213., 208., 88.),
    95: (64., 158., 70.),
    96: (18., 163., 194.),
    97: (65., 29., 153.),
    98: (177., 10., 109.),
    99: (152., 83., 7.),
    100: (83., 175., 30.),
    101: (18., 199., 153.),
    102: (61., 81., 208.),
    103: (213., 85., 216.),
    104: (170., 53., 42.),
    105: (161., 192., 38.),
    106: (23., 241., 91.),
    107: (12., 103., 170.),
    110: (151., 41., 245.),
    112: (133., 51., 80.),
    115: (184., 162., 91.),
    116: (50., 138., 38.),
    118: (31., 237., 236.),
    120: (39., 19., 208.),
    121: (223., 27., 180.),
    122: (254., 141., 85.),
    125: (97., 144., 39.),
    128: (106., 231., 176.),
    130: (12., 61., 162.),
    131: (124., 66., 140.),
    132: (137., 66., 73.),
    134: (250., 253., 26.),
    136: (55., 191., 73.),
    138: (60., 126., 146.),
    139: (153., 108., 234.),
    140: (184., 58., 125.),
    141: (135., 84., 14.),
    145: (139., 248., 91.),
    148: (53., 200., 172.),
    154: (63., 69., 134.),
    155: (190., 75., 186.),
    156: (127., 63., 52.),
    157: (141., 182., 25.),
    159: (56., 144., 89.),
    161: (64., 160., 250.),
    163: (182., 86., 245.),
    165: (139., 18., 53.),
    166: (134., 120., 54.),
    168: (49., 165., 42.),
    169: (51., 128., 133.),
    170: (44., 21., 163.),
    177: (232., 93., 193.),
    180: (176., 102., 54.),
    185: (116., 217., 17.),
    188: (54., 209., 150.),
    191: (60., 99., 204.),
    193: (129., 43., 144.),
    195: (252., 100., 106.),
    202: (187., 196., 73.),
    208: (13., 158., 40.),
    213: (52., 122., 152.),
    214: (128., 76., 202.),
    221: (187., 50., 115.),
    229: (180., 141., 71.),
    230: (77., 208., 35.),
    232: (72., 183., 168.),
    233: (97., 99., 203.),
    242: (172., 22., 158.),
    250: (155., 64., 40.),
    261: (118., 159., 30.),
    264: (69., 252., 148.),
    276: (45., 103., 173.),
    283: (111., 38., 149.),
    286: (184., 9., 49.),
    300: (188., 174., 67.),
    304: (53., 206., 53.),
    312: (97., 235., 252.),
    323: (66., 32., 182.),
    325: (236., 114., 195.),
    331: (241., 154., 83.),
    342: (133., 240., 52.),
    356: (16., 205., 144.),
    370: (75., 101., 198.),
    392: (237., 95., 251.),
    395: (191., 52., 49.),
    399: (227., 254., 54.),
    408: (49., 206., 87.),
    417: (48., 113., 150.),
    488: (125., 73., 182.),
    540: (229., 32., 114.),
    562: (158., 119., 28.),
    570: (60., 205., 27.),
    572: (18., 215., 201.),
    581: (79., 76., 153.),
    609: (134., 13., 116.),
    748: (192., 97., 63.),
    776: (108., 163., 18.),
    1156: (95., 220., 156.),
    1163: (98., 141., 208.),
    1164: (144., 19., 193.),
    1165: (166., 36., 57.),
    1166: (212., 202., 34.),
    1167: (23., 206., 34.),
    1168: (91., 211., 236.),
    1169: (79., 55., 137.),
    1170: (182., 19., 117.),
    1171: (134., 76., 14.),
    1172: (87., 185., 28.),
    1173: (82., 224., 187.),
    1174: (92., 110., 214.),
    1175: (168., 80., 171.),
    1176: (197., 63., 51.),
    1178: (175., 199., 77.),
    1179: (62., 180., 98.),
    1180: (8., 91., 150.),
    1181: (77., 15., 130.),
    1182: (154., 65., 96.),
    1183: (197., 152., 11.),
    1184: (59., 155., 45.),
    1185: (12., 147., 145.),
    1186: (54., 35., 219.),
    1187: (210., 73., 181.),
    1188: (221., 124., 77.),
    1189: (149., 214., 66.),
    1190: (72., 185., 134.),
    1191: (42., 94., 198.),
    1200: (0, 0, 0)
}

Replica_LABELS_label30 = ['wall', 'floor', 'cabinet', 'bed', "chair",
                           'sofa', 'table', 'door', 'window', 'countertop',
                           'desk', 'shelf', 'curtain', 'pillow', 'picture', 
                           'ceiling', 'refrigerator', 'tv', 'nightstand', 'toilet', 
                           'sink',  'bathtub',  'clock', 'backpack', 'trash can',
                           'stool', 'microwave', 'bicycle', 'pair of shoes', "computer", 
                           "other"]

MATTERPORT_LABELS_label35 = ['wall', 'floor', 'cabinet', 'bed', 'chair', 
                        'sofa', 'table', 'door', 'window', 'counter',
                        'desk', 'shelf', 'curtain', 'pillow', 'picture', 
                        'ceiling', 'refrigerator', 'tv', 'nightstand', 'toilet', 
                        'sink',  'bathtub',  'lamp', 'person', 'clothes', 
                        'book',  'mirror', 'backpack', 'trash can', 'plant',
                        'banister', 'stairs',  'stool', 'vase', "other"]

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
20 :( 112 , 128 , 144 ) , # sink
21 :( 153.0 , 108.0 , 234.0 ) ,# bathtub
22: (237.0, 204.0, 37.0), # lamp
23: (220, 20, 60), # person
24: (34.0, 14.0, 130.0), # clothes
25: (192.0, 229.0, 91.0), # book
26: (162.0, 62.0, 60.0), # mirror
27 :( 118.0 , 174.0 , 76.0 ) , #backpack
28 :( 82 , 84 , 163 ) , # trash can
29: (150.0, 53.0, 56.0), # plant
30: (64.0, 158.0, 70.0), # banister
31: (208.0, 49.0, 84.0), # stairs
32 :( 143.0 , 45.0 , 115.0 ) , #stool
33: ( 102.0 , 255.0 , 255.0 ) , # vase
34 :( 0 , 0 , 0 ) ,
}

Replica_COLOR_MAP_label30 = {
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
20 :( 112 , 128 , 144 ) , # sink
21 :( 153.0 , 108.0 , 234.0 ) ,# guitar
22 :( 102.0 , 255.0 , 255.0 ) , #clock
23 :( 118.0 , 174.0 , 76.0 ) , #backpack
24 :( 82 , 84 , 163 ) , # trash can
25 :( 143.0 , 45.0 , 115.0 ) , #stool
26 :( 177.0 , 82.0 , 239.0 ) , #microwave
27 :( 58.0 , 98.0 , 137.0 ) ,
28 :( 127.0 , 63.0 , 52.0 ) ,
29 :( 17.0 , 242.0 , 171.0 ) ,
30 :( 0 , 0 , 0 ) ,
}

Matterport_Evaluate_label35 = ['wall', 'floor', 'cabinet', 'bed', 'chair', 
                        'sofa', 'table', 'door', 'window', 'counter',
                        'desk', 'shelf', 'curtain', 'pillow', 'picture', 
                        'ceiling', 'refrigerator', 'tv', 'nightstand', 'toilet', 
                        'sink',  'bathtub',  'lamp', 'person', 'clothes', 
                        'book',  'mirror']

Replica_Evaluate_label30 = ['wall', 'floor', 'cabinet', 'bed', "chair",
                           'sofa', 'table', 'door', 'window', 'countertop',
                           'desk', 'shelf', 'curtain', 'pillow', 'picture', 
                           'ceiling', 'refrigerator', 'tv', 'nightstand', 'toilet', 
                           'sink',  'bathtub',  'clock', 'backpack', 'trash can',
                           ]
