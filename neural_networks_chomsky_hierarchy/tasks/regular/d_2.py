"""Compute whether the input string is D2."""

import functools
from typing import Mapping

from jax import numpy as jnp

from neural_networks_chomsky_hierarchy.tasks import task

import numpy as np
from collections import Counter
import logging


class D2(task.GeneralizationTask):
  """A task which goal is to check whether the input string is D2.

  The input is a binary string, composed of 0s and 1s. If they are (0(01)*1)*,
  the class is 1, otherwise it's 0.
  
  Most functions are adapted from:
  https://github.com/satwik77/Transformer-Formal-Languages/blob/main/src/utils/starfree_generator.py#L566
  """
  def __init__(self):
    self.sigma = ['0', '1']
    self.n = 2

  def random_select_length(self, maxlength, mean_ratio=0.75, std_ratio=0.1):
    mean = maxlength * mean_ratio
    std = std_ratio * mean
    length = int(std * np.random.randn() + mean)
    length = (length//2)*2

    return min(maxlength, length)

  def generate_d_n(self, n, maxlength):
    if n == 0 or maxlength == 0:
        return ''

    d_n = ''
    while len(d_n) < maxlength:
        length_d_n_min_1 = self.random_select_length(maxlength-len(d_n)-2)
        d_n_min_1 = self.generate_d_n(n-1, length_d_n_min_1)
        d_n += '0{}1'.format(d_n_min_1)

    return d_n

  def generate_string(self, length):
    return self.generate_d_n(self.n, length)

  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of strings and the expected class."""
    strings = []
    if length % 2 == 1:
      strings += self.generate_negatives(batch_size, length)
      ans = [0] * batch_size
    else:
      strings += self.generate_positives(batch_size//2, length)
      bad_strings = self.generate_negatives(batch_size//2, length)
      strings += bad_strings
      ans = [1] * (batch_size//2) + [0] * (batch_size//2)

    def get_one_hot(targets, num_classes):
      targets = np.array(targets)
      res = np.eye(num_classes)[targets.reshape(-1)]
      return res.reshape(list(targets.shape)+[num_classes])

    one_hot_strings = get_one_hot(strings, num_classes=2)
    ans = get_one_hot(ans, num_classes=self.output_size)

    return {
        'input': one_hot_strings,
        'output': ans,
    }
  
  def generate_positives(self, num, length):
    arr = []
    while len(arr) < num:
      string = self.generate_string(length)
      arr.append([int(s) for s in string])

    return arr
  
  def generate_negatives(self, num, length):
    arr = []
    while len(arr) < num:
      string = np.random.choice(self.sigma, size=length)
      if not self.belongs_to_lang(string):
          arr.append([int(s) for s in string])

    return arr
  
  def belongs_to_lang(self, string):
    string = ''.join(string)
    if len(string) % 2 == 1:
      return False
    else:
      depth = 0
      while len(string) and '01' in string:
          string = string.replace('01', '')
          depth += 1
      return (depth <= self.n) and (len(string) == 0)

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 2

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2
