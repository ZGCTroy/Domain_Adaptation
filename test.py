import torch
import numpy as np
scl = np.exp(
    np.random.uniform(
        low=np.log(0.75),
        high=np.log(1.33),
        size=(10,)
    )
)


print(scl)
