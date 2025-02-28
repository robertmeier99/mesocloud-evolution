"""
Useful functions for trajectory processing and cloudmetrics computation.
"""

import numpy as np

def where_both(condition_1,condition_2):
    return np.where(np.where(condition_1,True,False)*np.where(condition_2,True,False))