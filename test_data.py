import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
from ggplot import *
import pandas
import scipy
import scipy.stats
import pylab
from random import *


def create_df(length,n):
    df = pandas.DataFrame()
    df["a"] = ([randint(0,n) for x in range(0,length)])
    df["b"] = df["a"]*100
    print(df.head())
    return(df)

#print(create_df(1000,24))

d = create_df(10000000,24)

print(ggplot( d, aes("a",weight = "b")) + geom_bar())