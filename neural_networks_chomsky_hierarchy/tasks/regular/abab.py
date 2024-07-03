"""Compute whether the input string is (abab)*."""

import functools
from typing import Mapping

import jax
from jax import numpy as jnp

from neural_networks_chomsky_hierarchy.tasks import task

import numpy as np
import logging


class ABAB(task.GeneralizationTask):
  """A task which goal is to check whether the input string is (abab)*.

  The input is a binary string, composed of 0s and 1s. If they are (0101)*,
  the class is 0, otherwise it's one.

  Examples:
    0101 -> class 0
    01 -> class 1

  """

  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of strings and the expected class."""
    strings = []
    if length % 4 != 0:
      strings += self.generate_negatives(batch_size, length)
      ans = [0] * batch_size
    else:
      strings += self.generate_positives(batch_size//2, length)
      strings += self.generate_negatives(batch_size//2, length)
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
      string = '0101' * (length//4)
      arr.append([int(s) for s in string])

    return arr

  def generate_negatives(self, num, length):
    arr = []
    while len(arr) < num:
      string = np.random.choice(['0', '1'], size=length)
      if not self.belongs_to_lang(string):
        arr.append([int(s) for s in string])

    return arr

  def belongs_to_lang(self, string):
    string = ''.join(string)
    #logging.info('{} {} {}'.format(string, len(string), string.count('0101')))
    return (string.count('0101')*4) == len(string)

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 2

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2

