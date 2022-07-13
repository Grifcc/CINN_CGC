#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/7/12 14:19
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用
import paddle
import paddle.fluid as fluid
import paddle.nn as nn
import cinn
from cinn import *
from cinn.frontend import *
from cinn.framework import *
from cinn.common import *
import numpy as np
import os
import sys

model_dir = "./ResNet50"
input_tensor = 'inputs'
target_tensor = 'save_infer_model/scale_0.tmp_1'
x_shape = [1, 3, 224, 224]

x_var = paddle.uniform((2, 4, 8, 8), dtype='float32', min=-1., max=1.)
conv = nn.Conv2D(4, 6, (3, 3))
y_var = conv(x_var)
y_np = y_var.numpy()
print(y_np.shape)


if os.path.exists("is_cuda"):
    target = DefaultNVGPUTarget()
else:
    target = DefaultHostTarget()

params_combined = True
computation = Computation.compile_paddle_model(
    target, model_dir, [input_tensor], [x_shape], params_combined)

a_t = computation.get_tensor(input_tensor)
x_data = np.random.random(x_shape).astype("float32")
a_t.from_numpy(x_data, target)

out = computation.get_tensor(target_tensor)
out.from_numpy(np.zeros(out.shape(), dtype='float32'), target)

computation.execute()
res_cinn = out.numpy(target)
print("CINN Execution Done!")
