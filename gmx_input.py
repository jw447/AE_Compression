# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.

# The original gromacs simulation data has 33875 x 3 for single frame.
# To train a simple model, we set the input data at 100 x 3.

Data_Height = 1000
Data_Width = 3

# Global constants describing the CIFAR-10 data set.
# NUM_CLASSES = parse_binary_data.NUM_TAGS

# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 200*200*0.8
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 200*200*0.2
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL_ori = 2832

def read_data(filename_queue):
  """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.
    datatype: 'ori' or 'synthetic'
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  # TODO:
  # Consider add batch_size in the future

  class DataRecord(object):
    pass
  result = DataRecord()

  result.height = Data_Height # 1000
  result.width = Data_Width   # 3
  result.depth = 1

  reader = tf.TextLineReader()

  values = []
  for i in range(0,Data_Height):
    values.append(reader.read(filename_queue)[1])

  data = tf.decode_csv(values,[[1],[1.0],[1.0],[1.0]],field_delim=' ')
  # shape of data should equals [4,300]

  result.value = data[1:4]
  # result.value - list of 3 tensors with shape(300)

  return result
