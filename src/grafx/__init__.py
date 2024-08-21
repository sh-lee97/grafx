from . import data, draw, processors, render, utils

# USE_FLASHFFTCONV = True

try:
    profile  # throws an exception when profile isn't defined
except NameError:
    profile = lambda x: x  # if it's not defined simply ignore the decorator.
