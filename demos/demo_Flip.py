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

a = builder.create_input(Float(32), (2, 3, 4), "A")

e = builder.flip(a, 0)

computation = Computation.build_and_compile(target, builder)

A_data = np.array([
    [
        [1, 5, 5, 2],
        [9, -6, 2, 8],
        [-3, 7, -9, 1]
    ],
    [
        [-1, 7, -5, 2],
        [9, 6, 2, 8],
        [3, 7, 9, 1]
    ]]).astype("float32")

computation.get_tensor("A").from_numpy(A_data, target)
computation.execute()

e_tensor = computation.get_tensor(str(e))
a_tensor = computation.get_tensor(str(a))


edata_cinn = e_tensor.numpy(target)
adata_cinn = a_tensor.numpy(target)

print(adata_cinn.shape)
print(edata_cinn.shape)

print(adata_cinn)
print(edata_cinn)
