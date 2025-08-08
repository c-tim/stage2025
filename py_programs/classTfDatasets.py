#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 10:12:54 2025

@author: tim
"""
import tflearn
from tflearn.data_utils import load_csv
from classDatasets import Dataset

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class tfDatasets(Dataset):
    
    def __init__(self, path_file, module):
        super()
        module.download_dataset(path_file)
        self.inputs = load_csv(path_file, target_column=0,
                                categorical_labels=True, n_classes=2)
        self.train_inputs = self.inputs
        self.test_inputs = self.inputs

        '''        # Construct a tf.data.Dataset
        ds = tfds.load('mnist', split='train', shuffle_files=True)
        
        # Build your input pipeline
        ds = ds.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
        for example in ds.take(1):
          image, label = example["image"], example["label"]'''