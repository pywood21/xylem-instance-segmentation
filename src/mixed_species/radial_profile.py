import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy import ndimage
from tqdm import tqdm
from joblib import Parallel, delayed


def get_bbox(df, label_num):
    x = df[df['label']==label_num]['centroid-0'].values 
    y = df[df['label']==label_num]['centroid-1'].values 
    xw = df[df['label']==label_num]['axis_major_length'].values 
    yw = df[df['label']==label_num]['axis_minor_length'].values 
    return int(x-xw/2),int(x+xw/2),int(y-yw/2),int(y+yw/2)

def get_anatomy(df, label_num):
    anatomy = df[df['label']==label_num]['anatomy'].values 
    return anatomy

def show_example(df, cleared_label,label_num, img_f):
    tmp_label = (cleared_label == label_num)*1
    x,y = get_radial_profile(img_f,tmp_label)
    p,q,r,s = get_bbox(df, label_num)
    anatomy = get_anatomy(df, label_num)

    fig, ax = plt.subplots(1,2,figsize=(6,3))
    ax[0].scatter(x,y, s =3)
    ax[0].set_xlabel('distance from boundary (pixel)')
    ax[0].set_ylabel('gray level (mean)')
    ax[1].imshow(img_f[p:q,r:s]*tmp_label[p:q,r:s], cmap='Greens')
    ax[1].set_axis_off()
    fig.suptitle('cell number {cellnum}, category:{anatomy}'.format(cellnum = label_num, anatomy=anatomy[0]))
    plt.show()

def get_radial_profile(im_in, lbl_in):
    # Euclidean distance transform
    dt = ndimage.distance_transform_edt(lbl_in)
    unique_dt = np.unique(dt)
    r_profile = np.array([im_in[dt == d].mean() for d in unique_dt])
    return unique_dt, r_profile

def get_xy_profile(df, cleared_label, img_f):
    # ラベルとアナトミーのマッピングを事前に構築
    label_to_anatomy = dict(zip(df['label'], df['anatomy']))
    profiles = {"fiber": [], "vessel": [], "ray": [], "parenchyma": []}

    def process_label(label):
        anatomy = label_to_anatomy.get(label)
        if anatomy in profiles:
            tmp_label = (cleared_label == label).astype(np.int32)
            x, y = get_radial_profile(img_f, tmp_label)
            return anatomy, (x, y)
        return None

    # 並列処理
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_label)(label) for label in tqdm(df['label'].unique())
    )

    # 結果を整理
    for result in results:
        if result is not None:
            anatomy, profile = result
            profiles[anatomy].append(profile)

    return profiles["fiber"], profiles["vessel"], profiles["ray"], profiles["parenchyma"]


