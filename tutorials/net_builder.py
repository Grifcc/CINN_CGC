# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
"""
Run model by using NetBuilder API
=========================================================

In this tutorial, we will introduce the ways to build and run a model using NetBuilder APIs.
"""

import cinn
from cinn import frontend
from cinn import common
import numpy as np
# sphinx_gallery_thumbnail_path = './paddlepaddle.png'

##################################################################
#
# Define the NetBuilder.
# -----------------------------
#
# Using NetBuilder is a convenient way to build a model in CINN.
# You can build and run a model by invoking NetBuilder's API as following.
#
# :code:`name`: the ID of NetBuilder
#
# Generally, the API in `NetBuilder` is coarse-grained operator, in other words,
# the DL framework like Paddle's operator.
builder = frontend.NetBuilder(name="batchnorm_conv2d")

##################################################################
#
# Define the input variable of the model.
# ---------------------------------------------
#
# The input variable should be created by create_input API. Note that the variable
# here is just a placeholder, does not need the actual data.
#
# :code:`type`: the data type of input variable, now support `Void`, `Int`, `UInt`,
# `Float`, `Bool` and `String`, the parameter is the type's bit-widths, here the
# data type is `float32`.
#
# :code:`shape`: The shape of the input variable, note that here does not support
# dynamic shape, so the dimension value should be greater than 0 now.
#
# :code:`id_hint`: the name of variable, the defaule value is `""`
a = builder.create_input(
    type=common.Float(32), shape=(8, 3, 224, 224), id_hint="x")
scale = builder.create_input(type=common.Float(32), shape=[3], id_hint="scale")
bias = builder.create_input(type=common.Float(32), shape=[3], id_hint="bias")
mean = builder.create_input(type=common.Float(32), shape=[3], id_hint="mean")
variance = builder.create_input(
    type=common.Float(32), shape=[3], id_hint="variance")
weight = builder.create_input(
    type=common.Float(32), shape=(3, 3, 7, 7), id_hint="weight")

##################################################################
#
# Build the model by using NetBuilder API
# ---------------------------------------------
#
# For convenience, here we build a simple model that only consists of batchnorm and conv2d
# operators. Note that you can find the operator's detailed introduction in another
# document, we won't go into detail here.
y = builder.batchnorm(a, scale, bias, mean, variance, is_test=True)
res = builder.conv2d(y[0], weight)

##################################################################
#
# Set target
# ---------------------
#
# The target identified where the model should run, now we support
# two targets:
#
# :code:`DefaultHostTarget`: the model will running at cpu.
#
# :code:`DefaultNVGPUTarget`: the model will running at nv gpu.
if common.is_compiled_with_cuda():
    target = common.DefaultNVGPUTarget()
else:
    target = common.DefaultHostTarget()

print("Model running at ", target.arch)

##################################################################
#
# Generate the program
# ---------------------
#
# After the model building, the `Computation` will generate a CINN execution program,
# and you can get it like:
computation = frontend.Computation.build_and_compile(target, builder)

##################################################################
#
# Random fake input data
# -----------------------------
#
# Before running, you should read or generate some data to feed the model's input.
# :code:`get_tensor`: Get the tensor with specific name in computation.
# :code:`from_numpy`: Fill the tensor with numpy data.
tensor_data = [
    np.random.random([8, 3, 224, 224]).astype("float32"),  # a
    np.random.random([3]).astype("float32"),  # scale
    np.random.random([3]).astype("float32"),  # bias
    np.random.random([3]).astype("float32"),  # mean
    np.random.random([3]).astype("float32"),  # variance
    np.random.random([3, 3, 7, 7]).astype("float32")  # weight
]

computation.get_tensor("x").from_numpy(tensor_data[0], target)
computation.get_tensor("scale").from_numpy(tensor_data[1], target)
computation.get_tensor("bias").from_numpy(tensor_data[2], target)
computation.get_tensor("mean").from_numpy(tensor_data[3], target)
computation.get_tensor("variance").from_numpy(tensor_data[4], target)
computation.get_tensor("weight").from_numpy(tensor_data[5], target)

##################################################################
#
# Run program and print result
# -----------------------------
#
# Finally, you can run the model by invoking function `execute()`.
# After that, you can get the tensor you want by `get_tensor` with tensor's name.
computation.execute()
res_tensor = computation.get_tensor(str(res))
res_data = res_tensor.numpy(target)

# print result
print(res_data)
