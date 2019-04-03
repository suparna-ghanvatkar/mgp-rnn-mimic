'''
Script to create a list of patients splits which follows the various criteria..
The waveforms need to be analyzed from their dump file
'''
import pandas as pd
import numpy as np
from math import ceil, isnan
import pickle
import wfdb
import datetime
from collections import defaultdict

