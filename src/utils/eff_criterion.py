"""Nash efficiency criterion function"""

import numpy as np
import pandas as pd

def nse(predictions, targets):
    return (1-(np.sum((predictions-targets)**2)/np.sum((targets-np.mean(targets))**2)))
