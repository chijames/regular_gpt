"""Compute whether the input string is (aa)*."""

import functools
from typing import Mapping

import jax
import jax.nn as jnn
from jax import numpy as jnp

from neural_networks_chomsky_hierarchy.tasks import task


class AA(task.GeneralizationTask):
  """A task which goal is to check whether the input string is (aa)*.

  The input is a binary string, composed of 0s. If they are (00)*,
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
    ans = (length % 2)*jnp.ones(shape=(batch_size), dtype=jnp.int32)

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

