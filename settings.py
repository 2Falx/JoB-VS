# -*- coding: utf-8 -*-
import os
import math
import json
import numpy as np
import torch.nn as nn

# batch sizes used for training the network
precalculated_batch = {'0': 1} # [192, 192, 96]

def adjust_to_divisible_by_32(vector):
    print('adjust_to_divisible_by_32')
    adjusted_vector = []
    for value in vector:
        adjusted_value = (value // 32) * 32  # Round down to the nearest multiple of 32
        adjusted_vector.append(adjusted_value)
    return adjusted_vector

def plan_experiment(task, batch, patience, rank, model_name, root):
    root = os.path.join(root)
    
    with open(os.path.join(root, 'dataset_stats.json'), 'r') as f:
        dataset = json.load(f)
        mean_size = dataset['mean_size']
        small_size = dataset['small_size']
        volume = dataset['volume_small']
        modalities = len(dataset['modality'])
        classes = len(dataset['labels'])

    # Asuming the structures are cubes, if the volume of a structure is
    # smaller than 32^3 we should use only 4 downsamples
    threshold = 32 * 32 * 32
    num_downsamples = 4 if volume < threshold else 5

    # Analysis per axis
    tr_size, val_size, p_size, strides, size_last = calculate_sizes(num_downsamples, mean_size, small_size)
    
        
    print('tr_size: ', tr_size)
    print('val_size: ', val_size)
    print('p_size: ', p_size)

    # MEMORY CONSTRAINT 1
    # If the feature maps in the final stage are too big we make sure to use
    # five downsamples
    if size_last >= (12 * 12 * 6):
        print('--- Five downsamples ---')
        tr_size, val_size, p_size, strides, size_last = calculate_sizes(5, mean_size, small_size)
        num_downsamples = 5
        print('tr_size: ', tr_size)
        print('val_size: ', val_size)
        print('p_size: ', p_size)
        print('num_downsamples: ', num_downsamples)

    # MEMORY CONSTRAINT 2
    # If the feature maps in the final stage are still too big
    # we reduce the input size
    if size_last >= (12 * 12 * 4):
        print('--- Reducing input size ---')
        tr_size, val_size, p_size, strides, size_last = calculate_sizes(num_downsamples, mean_size, small_size, 6)
        print('tr_size: ', tr_size)
        print('val_size: ', val_size)
        print('p_size: ', p_size)
        
    feature_size = 48

    strides = list(map(list, zip(*strides)))
    if len(strides) == 4:
        strides.insert(0, [1, 1, 1])

    print('--- Training input (before) size set to {} ---'.format(tr_size))
    print('--- Validation input (before) size set to {} ---'.format(val_size))
    tr_size = adjust_to_divisible_by_32(tr_size)
    val_size = adjust_to_divisible_by_32(val_size)


    if rank == 0:
        print('Current task is {}, with {} modalities and {} classes'.format(
            task, modalities, classes))
        print('The mean size of the images in this dataset is:', mean_size)
        print('--- Training input size set to {} ---'.format(tr_size))
        print('--- Validation input size set to {} ---'.format(val_size))

    hyperparams = {
        'task': task,
        'classes': classes,
        'p_size': p_size,  # size of the patch before the transformations
        'in_size': tr_size,  # size of the patch that enters the network
        'val_size': val_size,  # size of the patch that enters the network
        'test_size': val_size,  # size of the patch that enters the network
        'batch': batch,
        'test_batch': int(batch * 1),
        'patience': patience,  # Make it dependent on the data?
        'seed': 12345,
        'output_folder': '',
        'root': os.path.join(root),
        'data_file': os.path.join(root, 'dataset.json'),
    }

    model = {
        'classes': classes,
        'modalities': modalities,
        'strides': strides[:3],
        'img_size': tr_size,
        'in_channels': modalities,
        'out_channels': classes,
        'feature_size': feature_size,
        'use_checkpoint': True,
    }
    return hyperparams, model


def calculate_sizes(num_downsamples, mean_size, small_size, max_pow=7):
    tr_size = []
    val_size = []
    strides = []
    p_size = []
    size_last = 1
    # Analysis per axis
    for i, j in zip(mean_size, small_size):
        # If the image is too big we'll treat it as if it was smaller to avoid
        # ending up with huge input patches (memory constraint)
        i_big = i
        if i > 128:
            i_big = i
            i *= 0.7

        # Calculate the maximum possible input patch size
        power_t = min(int(math.log(i, 2)), max_pow)  # Max 64 or 128
        power_v = min(int(math.log(i, 2)), 7)  # Max 128
        sz_t = pow(2, power_t)
        sz_v = pow(2, power_v)

        # Calculate the number of strides
        stride = min(min(int(math.log(j, 2)), num_downsamples), power_t - 2)
        temp = np.ones(num_downsamples, dtype=int) * 2
        temp[:-stride] = 1
        strides.append(list(temp))

        # Calculate the input patch size
        constraint = sz_t / (2 ** stride) >= 8 and num_downsamples == 5
        if sz_t * 1.5 < i and not constraint:
            sz_t *= 1.5  # Max 196
        if sz_v * 1.5 < i:
            sz_v *= 1.5  # Max 196
            
        tr_size.append(int(sz_t))
        val_size.append(np.maximum(int(i_big // 2), int(sz_v)))
        size_last *= (sz_t / (2 ** stride))

        if sz_t + 20 < i:
            p_size.append(int(sz_t + 20))
        else:
            p_size.append(int(sz_t))
            
    return tr_size, val_size, p_size, strides, size_last
