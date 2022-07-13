#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
@Time : 2022/7/10 12:22
@Author : 詹荣瑞
@File : squeeze.py
@desc : 本代码未经授权禁止商用
"""
import numpy as np
from cinn.frontend import *
from cinn import Target
from cinn.framework import *
from cinn import runtime
from cinn import ir
from cinn import lang
from cinn.common import *

# target = DefaultHostTarget()
target = DefaultNVGPUTarget()

builder = CinnBuilder("test_basic")
a = builder.create_input(Float(32), (1, 24, 16, 1, 16, 16), "A")
a = builder.add(a, a)
a = builder.add(a, a)
a8 = builder.add(a, a)
e = builder.squeeze(a8)
# e = builder.reshape(e, (144, 56, 56))
builder.build()
computation = Computation.build_and_compile(target, builder)

A_data = np.random.random([1, 24, 16, 1, 16, 16]).astype("float32")

a_tensor = computation.get_tensor("A")
a_tensor.from_numpy(A_data, target)
# print(computation.get_tensor("A").reshape)
computation.execute()

a_tensor = computation.get_tensor("A")
e_tensor = computation.get_tensor(str(e))

edata_cinn = e_tensor.numpy(target)
adata_cinn = a_tensor.numpy(target)
print(adata_cinn.shape)
print(edata_cinn.shape)
print(np.sum((adata_cinn.squeeze()*8-edata_cinn)**2))

# computation = Computation.compile_paddle_model(
#     target = target, model_dir = "./ResNet50", input_tensors = ['inputs'], input_sapes = [[1, 3, 224, 224]], params_combined = True)
#
# # Get input tensor and set input data
# a_tensor = computation.get_tensor(input_tensor).from_numpy(A_data, target)

