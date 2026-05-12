from joblib import Parallel, delayed
import numpy as np
import numba
from numba import cuda
from scipy import ndimage


def get_xy_profile(df, cleared_label, img_f):
    label_to_anatomy = dict(zip(df['label'], df['anatomy']))
    profiles = {"fiber": [], "vessel": [], "ray": [], "parenchyma": []}

    def process_label(label):
        anatomy = label_to_anatomy.get(label)
        if anatomy in profiles:
            tmp_label = (cleared_label == label).astype(np.int32)
            x, y = get_radial_profile_gpu(img_f, tmp_label)
            return anatomy, (x, y)
        return None

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_label)(label) for label in df['label'].unique()
    )

    for result in results:
        if result is not None:
            anatomy, profile = result
            profiles[anatomy].append(profile)

    return profiles["fiber"], profiles["vessel"], profiles["ray"], profiles["parenchyma"]


@cuda.jit
def compute_profile(image, dt, profile, counts):
    """GPUでラジアルプロファイルを計算"""
    x, y = cuda.grid(2)
    if x < image.shape[0] and y < image.shape[1]:
        d = dt[x, y]
        if d > 0:
            index = int(d)
            cuda.atomic.add(profile, index, image[x, y])
            cuda.atomic.add(counts, index, 1)


def get_radial_profile_gpu(im_in, lbl_in):
    # 距離変換
    dt = ndimage.distance_transform_edt(lbl_in).astype(np.int32)
    max_distance = int(dt.max())

    # 距離プロファイル用の配列
    profile = np.zeros(max_distance + 1, dtype=np.float32)
    counts = np.zeros(max_distance + 1, dtype=np.int32)

    # GPUメモリにデータ転送
    d_image = cuda.to_device(im_in)
    d_dt = cuda.to_device(dt)
    d_profile = cuda.to_device(profile)
    d_counts = cuda.to_device(counts)

    # CUDA設定
    threads_per_block = (16, 16)
    blocks_per_grid_x = (dt.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (dt.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # CUDAカーネル呼び出し
    compute_profile[blocks_per_grid, threads_per_block](d_image, d_dt, d_profile, d_counts)

    # デバイスからホストへ転送
    profile = d_profile.copy_to_host()
    counts = d_counts.copy_to_host()

    # 平均値計算
    valid_counts = counts > 0
    profile[valid_counts] /= counts[valid_counts]

    return np.arange(max_distance + 1)[valid_counts], profile[valid_counts]