"""Compute whether the input string is Tomita 6."""

import functools
from typing import Mapping

import jax
from jax import numpy as jnp

from neural_networks_chomsky_hierarchy.tasks import task

import numpy as np
from collections import Counter
import logging


class Tomita6(task.GeneralizationTask):
  """A task which goal is to check whether the input string is Tomita 7.

  The input is a binary string, composed of 0s and 1s.
  If the difference in the number of 1s and 0s is divisible by 3,
  the class is 1, otherwise it's 0.

  """
  def __init__(self):
    self.tomita6 = Tomita6Language()

  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of strings and the expected class."""
    strings = []
    if length == 1:
      strings += self.tomita6.generate_negatives(batch_size, length)
      ans = [0] * batch_size
    else:
      strings += self.tomita6.generate_positives(batch_size//2, length)
      strings += self.tomita6.generate_negatives(batch_size//2, length)
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

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 2

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2


class Tomita6Language:
  def __init__(self):
    self.sigma = [0, 1]

  def belongs_to_lang(self, seq):
    counter = Counter(seq)
    return abs(counter[0] - counter[1]) % 3 == 0

  def generate_positives(self, num, length):
    arr = []
    while len(arr) < num:
      string = np.random.choice(self.sigma, size=length)
      if self.belongs_to_lang(string):
        arr.append(string)
    return arr
  
  def generate_negatives(self, num, length):
    arr = []
    while len(arr) < num:
      string = np.random.choice(self.sigma, size=length)
      if not self.belongs_to_lang(string):
        arr.append(string)
    return arr
