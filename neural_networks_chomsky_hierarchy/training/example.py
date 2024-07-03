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

"""Example script to train and evaluate a network."""

import argparse

import haiku as hk
import jax.numpy as jnp
import numpy as np

from neural_networks_chomsky_hierarchy.training import constants
from neural_networks_chomsky_hierarchy.training import curriculum as curriculum_lib
from neural_networks_chomsky_hierarchy.training import training
from neural_networks_chomsky_hierarchy.training import utils
from neural_networks_chomsky_hierarchy.models import positional_encodings as pos_encs_lib

import logging
logging.basicConfig(
  format='%(asctime)s %(levelname)-8s %(message)s',
  level=logging.INFO,
  datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--mod", default=5, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--sequence_length", default=40, type=int)
parser.add_argument("--embedding_dim", default=64, type=int)
parser.add_argument("--num_heads", default=8, type=int)
parser.add_argument("--chunk_size", default=2, type=int)
parser.add_argument("--thickness", default=1, type=int)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--dropout_prob", default=0.0, type=float)
parser.add_argument("--steps", default=1_000_000, type=int)
parser.add_argument("--architecture", default="transformer_encoder", type=str)
parser.add_argument("--max_range_test_length", default=500, type=int)
parser.add_argument("--run_gradient_analysis", action='store_true')
args = parser.parse_args()
logging.info(args)

def main() -> None:
  # Change your hyperparameters here. See constants.py for possible tasks and
  # architectures.
  
  architecture_params = {
    'embedding_dim': args.embedding_dim,
    'dropout_prob': args.dropout_prob,
    'chunk_size': args.chunk_size,
    'thickness': args.thickness,
    'positional_encodings': None,
    'positional_encodings_params': None,
    'num_heads': args.num_heads,
    'share_weight': True,
    'use_front_rear_pos': False,
    'num_layers': None,
  }

  '''
  architecture_params = {
    'embedding_dim': args.embedding_dim,
    'dropout_prob': args.dropout_prob,
    'chunk_size': None,
    'positional_encodings': pos_encs_lib.PositionalEncodings.RELATIVE,
    'positional_encodings_params': pos_encs_lib.RelativeParams(),
    'num_heads': args.num_heads,
    'share_weight': False,
    'use_front_rear_pos': False,
    'num_layers': 5,
  }
  '''
  
  # Create the task.
  if args.run_gradient_analysis:
    curriculum = curriculum_lib.FixedCurriculum(args.sequence_length)
  else:
    curriculum = curriculum_lib.UniformCurriculum(
        values=list(range(1, args.sequence_length + 1)))
  if args.task == 'modular_arithmetic':
    task = constants.TASK_BUILDERS[args.task](args.mod)
  else:
    task = constants.TASK_BUILDERS[args.task]()

  # Create the model.
  is_autoregressive = False
  computation_steps_mult = 0
  single_output = task.output_length(10) == 1
  model = constants.MODEL_BUILDERS[args.architecture](
      output_size=task.output_size,
      run_gradient_analysis=args.run_gradient_analysis,
      **architecture_params)
  if is_autoregressive:
    if 'transformer' not in args.architecture:
      model = utils.make_model_with_targets_as_input(
          model, computation_steps_mult
      )
    model = utils.add_sampling_to_autoregressive_model(model, single_output)
  else:
    model = utils.make_model_with_empty_targets(
        model, task, args.chunk_size, computation_steps_mult, single_output
    )
  model = hk.transform(model)

  # Create the loss and accuracy based on the pointwise ones.
  def loss_fn(output, target):
    loss = jnp.mean(jnp.sum(task.pointwise_loss_fn(output, target), axis=-1))
    return loss, {}

  def accuracy_fn(output, target):
    mask = task.accuracy_mask(target)
    return jnp.sum(mask * task.accuracy_fn(output, target)) / jnp.sum(mask)

  # Create the final training parameters.
  training_params = training.ClassicTrainingParams(
      seed=0,
      model_init_seed=args.seed,
      training_steps=args.steps,
      log_frequency=100,
      length_curriculum=curriculum,
      batch_size=args.batch_size,
      task=task,
      model=model,
      loss_fn=loss_fn,
      learning_rate=args.lr,
      accuracy_fn=accuracy_fn,
      compute_full_range_test=True,
      max_range_test_length=args.max_range_test_length,
      range_test_total_batch_size=512,
      range_test_sub_batch_size=64,
      is_autoregressive=is_autoregressive,
      run_gradient_analysis=args.run_gradient_analysis)
  
  training_worker = training.TrainingWorker(training_params, use_tqdm=False)
  _, eval_results, _ = training_worker.run()

  # Gather results and print final score.
  accuracies = [r['accuracy'] for r in eval_results]
  score = np.mean(accuracies[args.sequence_length + 1:])
  print(f'Network score: {score}')

if __name__ == '__main__':
  main()
