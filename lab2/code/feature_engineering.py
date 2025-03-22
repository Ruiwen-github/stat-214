import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def make_data(path, patch_size=9):
    """
    Load the image data and create patches from it.
    Args:
        patch_size: The size of the patches to create.
    Returns:
        images_long: A list of numpy arrays of the original images.
        patches: A list of lists of patches for each image.
    """

    # load isolated labeled images
    filepaths = path
    images_long = []
    for fp in filepaths:
        npz_data = np.load(fp)
        key = list(npz_data.files)[0]
        data = npz_data[key]
        images_long.append(data)

    # Compute global min and max for x and y over all images
    all_y = np.concatenate([img[:, 0] for img in images_long]).astype(int)
    all_x = np.concatenate([img[:, 1] for img in images_long]).astype(int)
    global_miny, global_maxy = all_y.min(), all_y.max()
    global_minx, global_maxx = all_x.min(), all_x.max()
    height = int(global_maxy - global_miny + 1)
    width = int(global_maxx - global_minx + 1)

    # Reshape each image onto the common grid.
    nchannels = images_long[0].shape[1] - 2
    images = []
    for img in images_long:
        y = img[:, 0].astype(int)
        x = img[:, 1].astype(int)
        # Use global minimums to get relative coordinates.
        y_rel = y - global_miny
        x_rel = x - global_minx
        image = np.zeros((nchannels, height, width))
        valid_mask = (y_rel >= 0) & (y_rel < height) & (x_rel >= 0) & (x_rel < width)
        y_valid = y_rel[valid_mask]
        x_valid = x_rel[valid_mask]
        img_valid = img[valid_mask]
        for c in range(nchannels):
            image[c, y_valid, x_valid] = img_valid[:, c + 2]
        images.append(image)
    print('done reshaping images')

    # Now that all images have the same shape, convert to a 4D array.
    images = np.array(images)
    pad_len = patch_size // 2

    # Global normalization across images.
    means = np.mean(images, axis=(0, 2, 3))[:, None, None]
    stds = np.std(images, axis=(0, 2, 3))[:, None, None]
    images = (images - means) / stds

    patches = []
    for i in range(len(images_long)):
        if i % 10 == 0:
            print(f'working on image {i}')
        patches_img = []
        # Pad the image by reflecting across the border.
        img_mirror = np.pad(
            images[i],
            ((0, 0), (pad_len, pad_len), (pad_len, pad_len)),
            mode="reflect",
        )
        # Use global min values to compute relative indices.
        ys = images_long[i][:, 0].astype(int)
        xs = images_long[i][:, 1].astype(int)
        for y, x in zip(ys, xs):
            y_idx = int(y - global_miny + pad_len)
            x_idx = int(x - global_minx + pad_len)
            patch = img_mirror[
                :,
                y_idx - pad_len : y_idx + pad_len + 1,
                x_idx - pad_len : x_idx + pad_len + 1,
            ]
            patches_img.append(patch.astype(np.float64))
        patches.append(patches_img)

    return images_long, patches

def get_data(path):
    # Run make_data first
    images_long, patches = make_data(path, patch_size=9)

    # Corresponding names of each channel in order (excluding x, y, label columns)
    channel_names = ['NDAI', 'SD', 'CORR', 'DF', 'CF', 'BF', 'AF', 'AN']

    df_features = []

    for i in range(len(patches)):
        image_patches = patches[i]
        image_long = images_long[i]

        for j, patch in enumerate(image_patches):
            # Compute patch statistics (mean, std, min, max) for each channel
            means = np.mean(patch, axis=(1, 2))
            stds = np.std(patch, axis=(1, 2))
            mins = np.min(patch, axis=(1, 2))
            maxs = np.max(patch, axis=(1, 2))

            feature_row = {
                'y': image_long[j, 0],
                'x': image_long[j, 1],
                'expert_label': image_long[j, -1]
            }

            # Assign descriptive names per channel
            for c, name in enumerate(channel_names):
                feature_row[f'mean_{name}'] = means[c]
                feature_row[f'std_{name}'] = stds[c]
                feature_row[f'min_{name}'] = mins[c]
                feature_row[f'max_{name}'] = maxs[c]

            df_features.append(feature_row)

    # Convert to DataFrame
    df_features = pd.DataFrame(df_features)
    return df_features