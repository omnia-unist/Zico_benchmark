# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Runs the tf_cnn_benchmarks tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

from absl import app
from absl import flags as absl_flags
import tensorflow as tf

import all_reduce_benchmark_test
import allreduce_test
import benchmark_cnn_distributed_test
import benchmark_cnn_test
import cnn_util_test
import variable_mgr_util_test
from models import model_config

# Ideally, we wouldn't need this option, and run both distributed tests and non-
# distributed tests. But, TensorFlow allocates all the GPU memory by default, so
# the non-distributed tests allocate all the GPU memory. The distributed tests
# spawn processes that run TensorFlow, and cannot run if all the GPU memory is
# already allocated. If a non-distributed test is run, then a distributed test
# is run in the same process, the distributed test will fail because there is no
# more GPU memory for the spawned processes to allocate.
absl_flags.DEFINE_boolean('run_distributed_tests', False,
                          'If True, run the distributed tests. If False, the'
                          'non-distributed tests.')

absl_flags.DEFINE_boolean('full_tests', False,
                          'If True, all distributed or non-distributed tests '
                          'are run, which can take hours. If False, only a '
                          'subset of tests will be run. This subset runs much '
                          'faster and tests almost all the functionality as '
                          'the full set of tests, so it is recommended to keep '
                          'this option set to False.')

FLAGS = absl_flags.FLAGS


def main(_):
  loader = unittest.defaultTestLoader
  if FLAGS.full_tests:
    suite = unittest.TestSuite([
        loader.loadTestsFromModule(allreduce_test),
        loader.loadTestsFromModule(cnn_util_test),
        loader.loadTestsFromModule(variable_mgr_util_test),
        loader.loadTestsFromModule(benchmark_cnn_test),
        loader.loadTestsFromModule(all_reduce_benchmark_test),
    ])
    if model_config.can_import_contrib:
      from models.tf1_only import nasnet_test  # pylint: disable=g-import-not-at-top
      suite.addTest(loader.loadTestsFromModule(nasnet_test))
    dist_suite = unittest.TestSuite([
        loader.loadTestsFromModule(benchmark_cnn_distributed_test),
    ])
  else:
    suite = unittest.TestSuite([
        loader.loadTestsFromModule(allreduce_test),
        loader.loadTestsFromModule(cnn_util_test),
        loader.loadTestsFromModule(all_reduce_benchmark_test),
        loader.loadTestsFromModule(variable_mgr_util_test),
        loader.loadTestsFromTestCase(benchmark_cnn_test.TestAlexnetModel),
        loader.loadTestsFromTestCase(benchmark_cnn_test.TfCnnBenchmarksTest),
        loader.loadTestsFromTestCase(benchmark_cnn_test.VariableUpdateTest),
        loader.loadTestsFromTestCase(
            benchmark_cnn_test.VariableMgrLocalReplicatedTest),
    ])
    dist_suite = unittest.TestSuite([
        loader.loadTestsFromNames([
            'benchmark_cnn_distributed_test.DistributedVariableUpdateTest'
            '.testVarUpdateDefault',

            'benchmark_cnn_distributed_test.TfCnnBenchmarksDistributedTest'
            '.testParameterServer',
        ]),
    ])

  if FLAGS.run_distributed_tests:
    print('Running distributed tests')
    result = unittest.TextTestRunner(verbosity=2).run(dist_suite)
  else:
    print('Running non-distributed tests')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
  sys.exit(not result.wasSuccessful())


if __name__ == '__main__':
  app.run(main)
