# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Training loop for base generalization experiments."""

import dataclasses
import functools
import random
from typing import Tuple, List, Callable, Mapping, Optional, Any

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import logging

from neural_networks_chomsky_hierarchy.tasks import task as task_lib
from neural_networks_chomsky_hierarchy.training import curriculum as curriculum_lib
from neural_networks_chomsky_hierarchy.training import range_evaluation
from neural_networks_chomsky_hierarchy.training import utils

import numpy as onp

_LossMetrics = Optional[Mapping[str, jnp.ndarray]]
_LossFn = Callable[[chex.Array, chex.Array], Tuple[float, _LossMetrics]]
_AccuracyFn = Callable[[chex.Array, chex.Array], float]
_ModelApplyFn = Callable[..., chex.Array]
_MAX_RNGS_RESERVE = 50000


@dataclasses.dataclass
class ClassicTrainingParams:
  """Parameters needed to train classical architectures."""
  seed: int  # Used to sample during forward pass (e.g. from final logits).
  model_init_seed: int  # Used to initialize model parameters.
  training_steps: int
  log_frequency: int

  task: task_lib.GeneralizationTask
  length_curriculum: curriculum_lib.Curriculum
  batch_size: int

  model: hk.Transformed
  loss_fn: Callable[[jnp.ndarray, jnp.ndarray], Tuple[float, _LossMetrics]]
  learning_rate: float
  test_model: Optional[hk.Transformed] = None
  max_grad_norm: float = 1.
  is_autoregressive: bool = False

  compute_full_range_test: bool = False
  range_test_total_batch_size: int = 512
  range_test_sub_batch_size: int = 64
  max_range_test_length: int = 100
  run_gradient_analysis: bool = False

  accuracy_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray],
                                 jnp.ndarray]] = None


def _apply_loss_and_metrics_fn(
    params: hk.Params,
    rng_key: chex.PRNGKey,
    batch: task_lib.Batch,
    model_apply_fn: _ModelApplyFn,
    loss_fn: _LossFn,
    accuracy_fn: _AccuracyFn,
    is_autoregressive: bool = False,
) -> Tuple[float, Tuple[_LossMetrics, float]]:
  """Computes the model output and applies the loss function.

  Depending on whether a model is autoregressive or not, it will have a
  different number of input parameters (i.e., autoregressive models also require
  the targets as an input).

  Args:
    params: The model parameters.
    rng_key: The prng key to use for random number generation.
    batch: The data (consists of both inputs and outputs).
    model_apply_fn: The model function that converts inputs into outputs.
    loss_fn: A function that computes the loss for a batch of logits and labels.
    accuracy_fn: A function that computes the accuracy for a batch of logits and
      labels.
    is_autoregressive: Whether the model is autoregressive or not.

  Returns:
    The loss of the model for the batch of data, extra loss metrics and the
    accuracy, if accuracy_fn is not None.
  """
  if is_autoregressive:
    outputs = model_apply_fn(
        params, rng_key, batch["input"], batch["output"], sample=False)
  else:
    outputs = model_apply_fn(params, rng_key, batch["input"])

  loss, loss_metrics = loss_fn(outputs, batch["output"])
  if accuracy_fn is not None:
    accuracy = accuracy_fn(outputs, batch["output"])
  else:
    accuracy = None
  return loss, (loss_metrics, accuracy)

@functools.partial(
    jax.jit,
    static_argnames=(
        "model_apply_fn",
        "loss_fn",
        "accuracy_fn",
        "optimizer",
        "is_autoregressive",
        "run_gradient_analysis",
    ),
)
def _update_parameters(
    params: hk.Params,
    rng_key: chex.PRNGKey,
    batch: task_lib.Batch,
    model_apply_fn: _ModelApplyFn,
    loss_fn: _LossFn,
    accuracy_fn: _AccuracyFn,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    is_autoregressive: bool = False,
    run_gradient_analysis: bool = False,
) -> Tuple[hk.Params, optax.OptState, Tuple[float, _LossMetrics, float]]:
  """Applies a single SGD update step to the model parameters.

  Args:
    params: The model parameters.
    rng_key: The prng key to use for random number generation.
    batch: The data (consists of both inputs and outputs).
    model_apply_fn: The model function that converts inputs into outputs.
    loss_fn: A function that computes the loss for a batch of logits and labels.
    accuracy_fn: A function that computes the accuracy for a batch of logits and
      labels.
    optimizer: The optimizer that computes the updates from the gradients of the
      `loss_fn` with respect to the `params` and the previous `opt_state`.
    opt_state: The optimizer state, e.g., momentum for each variable when using
      Adam.
    is_autoregressive: Whether the model is autoregressive or not.
    run_gradient_analysis: Store l2 normalized gradients and exit.

  Returns:
    The updated parameters, the new optimizer state, and the loss, loss metrics
    and accuracy.
  """
  (loss, (metrics, accuracy)), grads = jax.value_and_grad(
      _apply_loss_and_metrics_fn,
      has_aux=True)(params, rng_key, batch, model_apply_fn, loss_fn,
                    accuracy_fn, is_autoregressive)
  if run_gradient_analysis:
    l2 = jnp.sqrt(jnp.sum(jnp.square(grads['transformer_encoder']['dummy']), axis=2))
    with open('l2s.npy', 'wb') as f:
      np.save(f, onp.array(l2))
    exit()

  updates, new_opt_state = optimizer.update(grads, opt_state)
  new_params = optax.apply_updates(params, updates)
  return new_params, new_opt_state, (loss, metrics, accuracy)


class TrainingWorker:
  """Training worker."""

  def __init__(self,
               training_params: ClassicTrainingParams,
               use_tqdm: bool = False):
    """Initializes the worker.

    Args:
      training_params: The training parameters.
      use_tqdm: Whether to add a progress bar to stdout.
    """
    self._training_params = training_params
    self._use_tqdm = use_tqdm

  def run(
      self
  ) -> Tuple[List[Mapping[str, Any]], Optional[List[Mapping[str, Any]]],
             chex.ArrayTree]:
    """Trains the model with the provided config.

    Returns:
      Results (various training and validation metrics), module parameters
      and router parameters.
    """
    training_params = self._training_params
    rngs_reserve = min(_MAX_RNGS_RESERVE, training_params.training_steps)

    random.seed(training_params.seed)
    np.random.seed(training_params.seed)
    rng_seq = hk.PRNGSequence(training_params.seed)
    rng_seq.reserve(rngs_reserve)

    results = []
    model = training_params.model
    task = training_params.task
    length_curriculum = training_params.length_curriculum
    #warmup_cosine_decay_scheduler = optax.warmup_cosine_decay_schedule(init_value=1e-5, peak_value=training_params.learning_rate,
    #                                                               warmup_steps=int(training_params.training_steps*0.1),
    #                                                               decay_steps=training_params.training_steps)
    optimizer = optax.chain(
        optax.clip_by_global_norm(training_params.max_grad_norm),
        optax.adam(learning_rate=training_params.learning_rate))
        #optax.adam(learning_rate=warmup_cosine_decay_scheduler))
    # set length to a number larger than the max evaluation length
    # otherwise haiku will complain about paramaters not being initialized
    if training_params.run_gradient_analysis:
      dummy_batch = task.sample_batch(
          next(rng_seq), length=training_params.length_curriculum._sequence_length,
          batch_size=training_params.batch_size)
    else:
      dummy_batch = task.sample_batch(
          next(rng_seq), length=5, batch_size=training_params.batch_size)
    model_init_rng_key = jax.random.PRNGKey(training_params.model_init_seed)

    if training_params.is_autoregressive:
      params = model.init(
          model_init_rng_key,
          dummy_batch["input"],
          dummy_batch["output"],
          sample=False)
    else:
      params = model.init(model_init_rng_key, dummy_batch["input"])
    
    def map_nested_fn(fn):
      '''Recursively apply `fn` to the key-value pairs of a nested dict'''
      def map_fn(nested_dict):
        return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
                for k, v in nested_dict.items()}
      return map_fn

    def get_frozen(k, v):
      if k.startswith('pos_'):
        return 'frozen'
      else:
        return 'trainable'

    label_fn = map_nested_fn(get_frozen)
    optimizer = optax.multi_transform({'trainable':optimizer, 'frozen':optax.set_to_zero()}, label_fn)
    opt_state = optimizer.init(params)
    self._params, self._step = params, 0
    
    if training_params.run_gradient_analysis:
      # overwrite init params using trained ones
      trained_params = utils.load('checkpoint')
      if 'transformer_encoder' not in trained_params:
        trained_params['transformer_encoder'] = {}
      trained_params['transformer_encoder']['dummy'] = params['transformer_encoder']['dummy']
      params = trained_params

    steps = range(training_params.training_steps + 1)
    if self._use_tqdm:
      steps = tqdm.tqdm(steps)
    for step in steps:
      # Randomness handled by either python.random or numpy.
      length = length_curriculum.sample_sequence_length(step)
      # Randomness handled by either jax, python.random or numpy.
      train_batch = task.sample_batch(
          next(rng_seq), length=length, batch_size=training_params.batch_size)
      
      if training_params.run_gradient_analysis:
        with open('input_ids.npy', 'wb') as f:
          np.save(f, np.array(train_batch['input']))

      params, opt_state, (
          train_loss, train_metrics, train_accuracy) = _update_parameters(
              params=params,
              rng_key=next(rng_seq),
              batch=train_batch,
              model_apply_fn=model.apply,
              loss_fn=training_params.loss_fn,
              accuracy_fn=training_params.accuracy_fn,
              optimizer=optimizer,
              opt_state=opt_state,
              is_autoregressive=training_params.is_autoregressive,
              run_gradient_analysis=training_params.run_gradient_analysis,
              )
      self._params, self._step = params, step
      #steps.set_postfix(loss=float(train_loss), accuracy=float(train_accuracy))
      logging.info('progress: {:.2f}, loss: {}, acc: {}'.format(step/training_params.training_steps, float(train_loss), float(train_accuracy)))

      log_freq = training_params.log_frequency
      if (log_freq > 0) and (step % log_freq == 0):
        log_data = {
            "step": step,
            "train_loss": float(train_loss),
        }
        if training_params.accuracy_fn is not None:
          log_data["train_accuracy"] = float(train_accuracy)
        for key, value in train_metrics.items():
          log_data[".".join(["train_metrics", key])] = np.array(value)
        results.append(log_data)

      # We need to access this private attribute since the default reserve size
      # can not be edited yet.
      if not rng_seq._subkeys:  # pylint: disable=protected-access
        rng_seq.reserve(rngs_reserve)

    #utils.save('checkpoint', params)

    eval_results = None
    if training_params.compute_full_range_test:
      eval_params = range_evaluation.EvaluationParams(
          model=training_params.test_model or model,
          params=params,
          accuracy_fn=training_params.accuracy_fn,
          sample_batch=task.sample_batch,
          max_test_length=training_params.max_range_test_length,
          total_batch_size=training_params.range_test_total_batch_size,
          sub_batch_size=training_params.range_test_sub_batch_size,
          is_autoregressive=training_params.is_autoregressive,
      )
      eval_results = range_evaluation.range_evaluation(
          eval_params, use_tqdm=False)

    return results, eval_results, params
