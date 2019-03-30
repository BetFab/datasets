# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
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

"""Video feature."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import shutil
import subprocess
import tempfile

import numpy as np
import six
import tensorflow as tf
from tensorflow_datasets.core.features import image_feature
from tensorflow_datasets.core.features import sequence_feature


class Video(sequence_feature.Sequence):
  """`FeatureConnector` for videos, encoding frames individually on disk.

  Video: The image connector accepts as input either a 4 dimensional uint8 array
  representing a video, a path or a file object that can be decoded with ffmpeg.
  Note that not all formats in ffmpeg support reading from pipes, so if the file
  is not located in the file system, it has to be first copied. For this, we
  provide the flag `copy_locally` which will copy the video to a temporary local
  file before passing it to ffpeg.

  Output:
    video: tf.Tensor of type tf.uint8 and shape
      [num_frames, height, width, channels], where channels must be 1 or 3

  Example:
    * In the DatasetInfo object:
        features=features.FeatureDict({
            'video': features.Video(shape=(None, 64, 64, 3)),
        })

    * During generation:
        yield {
            'input': np.ones(shape=(128, 64, 64, 3), dtype=np.uint8),
        }
      or
        yield {
              'input': '/path/to/video.avi',
        }
      or
        yield {
              'input': gfile.GFile('/complex/path/video.avi'),
        }
  """

  def __init__(self, shape, encoding_format='png', copy_locally=False,
               ffmpeg_extra_args=None):
    """Construct the connector.

    Args:
      shape: tuple of ints, the shape of the video (num_frames, height, width,
        channels), where channels is 1 or 3.
      encoding_format: The video is stored as a sequence of encoded images.
        You can use any encoding format supported by image_feature.Feature.
      copy_locally: If set and have to decode a path, the file will be first
        temporarilly copied before being passed on to ffmpeg.
      ffmpeg_extra_args: A list of additional args to be passed to the ffmpeg
        binary. Specifically, ffmpeg will be called as:

           ffmpeg -i <input_file> <ffmpeg_extra_args> %010d.<encoding_format>

    Raises:
      ValueError: If the shape is invalid
    """
    shape = tuple(shape)
    if len(shape) != 4:
      raise ValueError('Video shape should be of rank 4')
    self._copy_locally = copy_locally
    self._encoding_format = encoding_format
    self._extra_ffmpeg_args = list(ffmpeg_extra_args or [])
    super(Video, self).__init__(
        image_feature.Image(shape=shape[1:], encoding_format=encoding_format),
        length=shape[0],
    )

  def _ffmpeg_decode(self, path_or_fobj):
    ffmpeg_path = 'ffmpeg'

    if isinstance(path_or_fobj, six.string_types):
      ffmpeg_args = [ffmpeg_path, '-i', path_or_fobj]
      ffmpeg_stdin = None
    else:
      ffmpeg_args = [ffmpeg_path, '-i', 'pipe:0']
      path_or_fobj.seek(0)
      ffmpeg_stdin = path_or_fobj.read()

    ffmpeg_dir = tempfile.mkdtemp()
    output_pattern = os.path.join(ffmpeg_dir, '%010d.' + self._encoding_format)
    ffmpeg_args += self._extra_ffmpeg_args
    ffmpeg_args.append(output_pattern)
    try:
      process = subprocess.Popen(ffmpeg_args,
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
      stdout_data, stderr_data = process.communicate(ffmpeg_stdin)
      ffmpeg_ret_code = process.returncode
      if ffmpeg_ret_code:
        raise ValueError(
            'ffmpeg returned error code {}, command={}\n'
            'stdout={}\nstderr={}\n'.format(ffmpeg_ret_code,
                                            ' '.join(ffmpeg_args),
                                            stdout_data,
                                            stderr_data))
      images = []
      for image_path in sorted(os.listdir(ffmpeg_dir)):
        with open(os.path.join(ffmpeg_dir, image_path), 'rb') as frame_file:
          images.append(six.BytesIO(frame_file.read()))
      return images
    finally:
      shutil.rmtree(ffmpeg_dir)

  def encode_example(self, video_or_path_or_fobj):
    """Convert the given image into a dict convertible to tf example."""
    if isinstance(video_or_path_or_fobj, np.ndarray):
      encoded_video = video_or_path_or_fobj
    elif (isinstance(video_or_path_or_fobj, six.string_types) and
          self._copy_locally):
      video_temp_path = tempfile.mktemp()
      try:
        tf.gfile.Copy(video_or_path_or_fobj, video_temp_path)
        encoded_video = self._ffmpeg_decode(video_temp_path)
      finally:
        os.unlink(video_temp_path)
    else:
      encoded_video = self._ffmpeg_decode(video_or_path_or_fobj)
    return super(Video, self).encode_example(encoded_video)
