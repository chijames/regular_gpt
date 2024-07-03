"""Compute whether the input string is (aaaa)*."""

import functools
from typing import Mapping

import jax
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jrandom

from neural_networks_chomsky_hierarchy.tasks import task


class AAAA(task.GeneralizationTask):
  """A task which goal is to check whether the input string is (aaaa)*.

  The input is a binary string, composed of 0s. If they are (0000)*,
  the class is 0, otherwise it's one.

  Examples:
    0000 -> class 0
    0 -> class 1

  Note the sampling is jittable so this task is fast.
  """

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of strings and the expected class."""
    strings = jnp.zeros(shape=(batch_size, length), dtype=jnp.int32)
    one_hot_strings = jnn.one_hot(strings, num_classes=2)
    ans = int((length % 4) != 0)*jnp.ones(shape=(batch_size), dtype=jnp.int32)

    return {
        'input': one_hot_strings,
        'output': jnn.one_hot(ans, num_classes=self.output_size),
    }

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 2

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2

