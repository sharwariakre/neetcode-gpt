import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        new_z = np.subtract(z, np.max(z))
        s = sum(np.exp(new_z))
        return np.round(np.exp(new_z)/s,4)
