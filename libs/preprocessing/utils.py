import os
# import pdb
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

import warnings
warnings.filterwarnings("ignore")


def preprocess(monai_id, im, args, lb=None, partition='Tr'):
    print(f'Processing Patient: {im}')
    # =========================== Create OASIS folder ============================ #
    im = '/'.join(im.split('/')[1:])
    if lb is not None:
        lb = '/'.join(lb.split('/')[1:])
        mk_dir(os.path.join(args['path'], f'labels{partition}'))
    
    mk_dir(os.path.join(args['path'], f'images{partition}')) 
    # =========================== Load the image ============================ #
    image = nib.load(os.path.join(args['task'], im))
    voxel_size = np.asarray(image.header.get_zooms())[:3]  # Image spacing
    im_type = image.get_data_dtype()
    image = image.get_fdata()
    in_shape = image.shape[:3]

    # ======================== Interpolation details ======================== #
    spacing = args['spacing']  # New dataset spacing
    factor = voxel_size / spacing
    if (max(voxel_size) / min(voxel_size)) > 3:
        split = True
        axis = np.where(max(voxel_size) / voxel_size == 1)[0]
        axis = int(axis)
    elif (max(spacing) / min(spacing)) > 3:
        split = True
        axis = np.where(max(spacing) / spacing == 1)[0]
        axis = int(axis)
    else:
        split = False
        axis = 2  # Default value

    # =============== Processing the labels (if we have them) =============== #
    label = None
    if lb is not None:
        label = nib.load(os.path.join(args['task'], lb)).get_fdata() # dtype=float64
        label = label.astype(np.uint8) # dtype=uint8

    # ======================== Processing the images ======================== #
    # If the image has more than one modality
    if len(image.shape) > 3:
        image = np.split(image, image.shape[-1], axis=3)
        final_im = []
        if lb is not None:
            label = interpolate(label, factor, 0, split, axis)
        for idx, x in enumerate(image):
            x = interpolate(np.squeeze(x), factor, 3, split, axis)
            limits = [args['limits'][0][idx], args['limits'][1][idx]]
            stats = [args['stats'][0][idx], args['stats'][1][idx]]
            x = normalize(x, limits, stats, args['CT'])

            if lb is not None:
                assert x.shape == label.shape
            final_im.append(x)

        image = np.stack(final_im, axis=3)
        save_image(image, os.path.join(args['path'], f'images{partition}', monai_id),
                   args['affine'], im_type)
        if lb is not None:
            save_image(label, os.path.join(args['path'], f'labels{partition}', monai_id),
                       args['affine'], np.uint8)
            
    else: # If the image has only one modality
        image, label = cut_image(image, label, 300, axis)
        image = interpolate(image, factor, 3, split, axis)
        if lb is not None:
            label = interpolate(label, factor, 0, split, axis)
        # Just in clase the interpolation made it bigger again
        image, label = cut_image(image, label, 300, axis)
        image = normalize(image, args['limits'], args['stats'], args['CT'])
        if lb is not None:
            save_image(label, os.path.join(args['path'], f'labels{partition}', monai_id),
                       args['affine'], np.uint8)
            assert image.shape == label.shape
        save_image(image, os.path.join(args['path'], f'images{partition}', monai_id),
                   args['affine'], im_type)

        
    print(f'Patient {im} processed. Original shape: {in_shape}. Final shape: {image.shape[:3]}')


def normalize(im, limits, stats, CT):
    if not CT:
        im = (im - im.min()) / (im.max() - im.min())
    im = np.clip(im, limits[0], limits[1])
    im = (im - stats[0]) / stats[1]
    return im


def cut_image(im, lb, size, axis):
    # If the image has too many slices, we cut it making sure that all the
    # foreground voxels are contained.
    if lb is not None:
        if len(im.shape) == 3 and lb.shape[axis] > size:
            zmin, zmax = z_coords(lb > 0, axis)
            if (zmax - zmin) > size:
                min_cut = zmin
                max_cut = zmax
            else:
                center = zmin + (zmax - zmin) // 2
                min_cut = center - size // 2
                max_cut = center + size // 2

                if min_cut < 0:
                    max_cut += min_cut * (-1)
                    min_cut += min_cut * (-1)
                elif max_cut > lb.shape[axis]:
                    min_cut -= max_cut - lb.shape[axis]
                    max_cut -= max_cut - lb.shape[axis]

            if axis == 0:
                im = im[min_cut:max_cut, :, :]
                lb = lb[min_cut:max_cut, :, :]
            elif axis == 1:
                im = im[:, min_cut:max_cut, :]
                lb = lb[:, min_cut:max_cut, :]
            elif axis == 2:
                im = im[:, :, min_cut:max_cut]
                lb = lb[:, :, min_cut:max_cut]
    return im, lb


def z_coords(im, axis):
    all_axis = (0, 1, 2)
    normal_axis = all_axis[:axis] + all_axis[axis + 1:]
    z = np.any(im, axis=normal_axis)
    zmin, zmax = np.where(z)[0][[0, -1]]
    zmin = np.maximum(zmin - 30, 0)
    zmax = np.minimum(zmax + 30, im.shape[axis])
    return zmin, zmax


def interpolate(im, factor, order, split, axis):
    scale = factor.copy()
    if any(scale != 1):
        # if too anisotropic, we interpolate every slice separately and then
        # perform nearest neighbour over the Z axis
        if split:
            im = slice_interpolation(im, scale[np.arange(len(scale)) != axis],
                                     axis, order=order)
            scale[np.arange(len(scale)) != axis] = 1
            order = 0

        im = zoom(im, scale, order=order)
    return im


def slice_interpolation(im, factor, axis, order):
    slices = np.split(im, im.shape[axis], axis=axis)
    new = []
    for i in slices:
        new.append(zoom(np.squeeze(i), factor, order=order))
    new = np.stack(new, axis=axis)
    return new


def save_image(data, out_path, affine, dtype):
    new_data = nib.Nifti1Image(data, affine)
    new_data.set_data_dtype(dtype)
    nib.save(new_data, out_path)


def read_json_from_root(root):
    with open(os.path.join(root, f'dataset.json'), 'r') as f:
        dataset = json.load(f)
        numTraining = len(dataset['training'])
        numTest = len(dataset['validation'])
        CT = dataset['modality']['0'] == 'CT'

    with open(os.path.join(root, 'stats.json'), 'r') as f:
        statistics = json.load(f)
        low = statistics['0.5']
        high = statistics['99.5']
        mean = statistics['mean']
        std = statistics['std']
        spacing = statistics['spacing']

    print(f'training {numTraining}, testing {numTest}')
    return dataset, [low, high], [mean, std], spacing, CT

def read_json(task, root):
    # ! Modified to read annotations in OASIS format. [N. Valderrama ~ May 17th]
    with open(os.path.join(root, f'data_{task}.json'), 'r') as f:
        dataset = json.load(f)
        numTraining = len(dataset['training'])
        numTest = len(dataset['validation'])
        CT = dataset['modality']['0'] == 'CT'

    with open(os.path.join(root, 'stats.json'), 'r') as f:
        statistics = json.load(f)
        low = statistics[task]['0.5']
        high = statistics[task]['99.5']
        mean = statistics[task]['mean']
        std = statistics[task]['std']
        spacing = statistics[task]['spacing']

    print('Procesing task {}: training {}, testing {}'.format(
        task, numTraining, numTest))
    return dataset, [low, high], [mean, std], spacing, CT


def cases_list(dataset, path, folder, set, remake=False):
    # ! Modified to read annotations in OASIS format. [N. Valderrama ~ May 18th]
    # ! Modified to read monai names in previous format. [N. Valderrama ~ Jul 8th]
    """
    Check which cases are already calculated to avoid preprocessing
    those again
    """
    missing = []
    for i in dataset[set]:
        label = os.path.isfile(os.path.join(path, i['label'].split('/')[0], i['monai_name']))
        image = os.path.isfile(os.path.join(path, i['image'].split('/')[0], i['monai_name']))
        name = i['image']
        
        if label and image and not remake:
            print('File {} already processed'.format(name))
            continue

        missing.append(i)
    return missing

def mk_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
