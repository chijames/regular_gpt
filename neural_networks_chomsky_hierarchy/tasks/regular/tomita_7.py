"""Compute whether the input string is Tomita 7."""

import functools
from typing import Mapping

import jax
from jax import numpy as jnp

from neural_networks_chomsky_hierarchy.tasks import task

import numpy as np
from collections import Counter
import logging


class Tomita7(task.GeneralizationTask):
  """A task which goal is to check whether the input string is 0*1*0*1*.

  The input is a binary string, composed of 0s and 1s. If they are of Tomita 7,
  the class is 1, otherwise it's 0.

  Note the sampling is jittable so this task is fast.
  """
  def __init__(self):
    self.tomita7 = Tomita7Language()

  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of strings and the expected class."""
    strings = []
    if length <= 4:
      strings += self.tomita7.generate_positives(batch_size, length)
      ans = [1] * batch_size
    else:
      strings += self.tomita7.generate_positives(batch_size//2, length)
      strings += self.tomita7.generate_negatives(batch_size//2, length)
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


class DFA(object):
  def __init__(self, sigma, Q, delta, q0, F):
    self.sigma = sigma
    self.Q = Q
    self.delta = delta
    self.q0 = q0
    self.F = F

  def __call__(self, string):
    qt = self.q0
    for symbol in string:
        qt = self.delta(qt, symbol)
    return qt in self.F


class Tomita7Language:
  def __init__(self):
    self.sigma = ['0', '1']
    self.Q = ['q0', 'q1', 'q2', 'q3', 'q4']
    self.delta = self.transition_function
    self.q0 = 'q0'
    self.F = {'q0', 'q1', 'q2', 'q3'}
    self.dead_states = {'q4'}
    self.dfa = DFA(self.sigma, self.Q, self.delta, self.q0, self.F)

  def belongs_to_lang(self, seq):
    return self.dfa(seq)

  def check_string(self, string, length):
    if len(string) == length:
        return True
    else:
        return False

  def transition_function(self, q, s):
    if q == 'q0':
      if s == '0':
        return 'q0'
      if s == '1':
        return 'q1'
    if q == 'q1':
      if s == '0':
        return 'q2'
      if s == '1':
        return 'q1'
    if q == 'q2':
      if s== '0':
        return 'q2'
      if s == '1':
        return 'q3'
    if q == 'q3':
      if s == '0':
        return 'q4'
      else:
        return 'q3'
    if q == 'q4':
      return 'q4'

  def generate_string(self, length):
    string = ''
    num_zeros = np.random.randint(0, length+1)
    string += ''.join(['0' for _ in range(num_zeros)])
    if self.check_string(string, length):
      return string
    num_ones = np.random.randint(0, length - len(string) + 1)
    string += ''.join(['1' for _ in range(num_ones)])
    if self.check_string(string, length):
      return string
    num_zeros = np.random.randint(0, length - len(string) + 1)
    string += ''.join(['0' for _ in range(num_zeros)])
    if self.check_string(string, length):
      return string
    num_ones = length - len(string)
    string += ''.join(['1' for _ in range(num_ones)])
    return string
  
  def generate_positives(self, num, length):
    arr = []
    while len(arr) < num:
      string = self.generate_string(length)
      if self.belongs_to_lang(string):
        arr.append([int(s) for s in string])
    return arr
  
  def generate_negatives(self, num, length):
    arr = []
    while len(arr) < num:
      string = np.random.choice(self.sigma, size=length)
      if not self.belongs_to_lang(string):
        arr.append([int(s) for s in string])
    return arr
