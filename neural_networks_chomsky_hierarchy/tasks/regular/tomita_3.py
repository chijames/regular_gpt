"""Compute whether the input string is Tomita 3."""

import functools
from typing import Mapping

import jax
from jax import numpy as jnp

from neural_networks_chomsky_hierarchy.tasks import task

import numpy as np
from collections import Counter
from abc import ABC, abstractmethod
import logging


class Tomita3(task.GeneralizationTask):
  """A task which goal is to check whether the input string is (01)*.

  The input is a binary string, composed of 0s and 1s. If they are (01)*,
  the class is 1, otherwise it's 0.

  Examples:
    0101 -> class 1
    01010 -> class 0

  Note the sampling is jittable so this task is fast.
  """
  def __init__(self):
    self.tomita3 = Tomita3Language()

  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of strings and the expected class."""
    strings = []
    if length == 1:
      strings += self.tomita3.generate_positives(batch_size, length)
      ans = [1] * batch_size
    else:
      strings += self.tomita3.generate_positives(batch_size//2, length)
      strings += self.tomita3.generate_negatives(batch_size//2, length)
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
    if qt in self.F:
        return True
    else:
        return False


class Tomita3Language:
  def __init__(self):
    super(Tomita3Language, self).__init__()
    self.sigma = ['0', '1']
    self.Q = ['q0', 'q1', 'q2', 'q3', 'q4']
    self.delta = self.transition_function
    self.q0 = 'q0'
    self.F = {'q0', 'q1', 'q2'}
    self.dead_states = {'q3','q4'}
    self.dfa = DFA(self.sigma, self.Q, self.delta, self.q0, self.F)

  def belongs_to_lang(self, seq):
    return self.dfa(seq)

  def transition_function(self, q, s):
    if q == 'q0':
      if s == '0':
        return 'q0'
      if s == '1':
        return 'q1'
    if q == 'q1':
      if s == '0':
        return 'q3'
      if s == '1':
        return 'q0'
    if q == 'q2':
      if s== '0':
        return 'q3'
      if s == '1':
        return 'q1'
    if q == 'q3':
      if s == '0':
        return 'q2'
      if s == '1':
        return 'q4'
    if q == 'q4':
      return 'q4'

  def generate_string(self, length):
    string = ''
    last_toss = None
    last_one_count = 0
    while len(string) != length:
      toss = np.random.choice(['0', '1'])
      if toss == '1':
        char_count = np.random.randint(length - len(string) + 1)
        string += ''.join([toss for _ in range(char_count)])
        if last_toss == '0' and char_count != 0:
          last_one_count = char_count
        else:
          last_one_count += char_count
      else:
        if last_toss is None or last_one_count%2 == 0:
          char_count = np.random.randint(length - len(string) + 1)
          string += ''.join([toss for _ in range(char_count)])
        else:
          choices = np.arange(0, length - len(string) + 1, 2)
          char_count = np.random.choice(choices)
          string += ''.join([toss for _ in range(char_count)])
      if char_count != 0:
        last_toss = toss
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
