import numpy as np
from cinn.frontend import *
from cinn import Target
from cinn.framework import *
from cinn import runtime
from cinn import ir
from cinn import lang
from cinn.common import *

target = DefaultHostTarget()
# target = DefaultNVGPUTarget()

builder = CinnBuilder("test_basic")
a = builder.create_input(Float(32), (1, 24, 56, 56), "A")
b = builder.create_input(Float(32), (1, 24, 56, 56), "B")
c = builder.add(a, b)
d = builder.create_input(Float(32), (144, 24, 1, 1), "D")
e = builder.conv(c, d)  # c = conv(a+b, d)
e = builder.squeeze(e)
# e = builder.reshape(e, (144, 56, 56))
computation = Computation.build_and_compile(target, builder)

A_data = np.random.random([1, 24, 56, 56]).astype("float32")
B_data = np.random.random([1, 24, 56, 56]).astype("float32")
D_data = np.random.random([144, 24, 1, 1]).astype("float32")

computation.get_tensor("A").from_numpy(A_data, target)
computation.get_tensor("B").from_numpy(B_data, target)
computation.get_tensor("D").from_numpy(D_data, target)
# print(computation.get_tensor("A").reshape)
computation.execute()

e_tensor = computation.get_tensor(str(e))

edata_cinn = e_tensor.numpy(target)
print(edata_cinn.shape)
