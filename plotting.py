
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
from ggplot import *
import pandas
import scipy
import scipy.stats
import pylab



df = pandas.DataFrame.from_csv("turnstile_weather_v2.csv")
df2 = pandas.DataFrame.from_csv("turnstile_data_master_with_weather.csv")

#print (df.head())

my_df = df2.dropna().reset_index(drop=True)


def mann_whitney_plus_means(turnstile_weather):
    
    rain,no_rain = [turnstile_weather["ENTRIESn_hourly"][turnstile_weather["rain"]==x] for x in [1,0]]
    
    with_rain_mean = np.mean(rain)
    without_rain_mean = np.mean(no_rain)
    U,p = scipy.stats.mannwhitneyu(rain,no_rain)
    
    return with_rain_mean, without_rain_mean, U, p, with_rain_mean/without_rain_mean

#print (my_df.head())
    
def draw_histograms(turnstile_weather):
   # print(ggplot(aes(x='date', y='beef'), data=meat) + geom_line() + stat_smooth(colour='blue', span=0.2))
    # rain,no_rain = [turnstile_weather["ENTRIESn_hourly"][turnstile_weather["rain"]==x] for x in [1,0]]
     #print(rain)
   # pt =  ggplot(turnstile_weather, aes(x="ENTRIESn_hourly")) + geom_histogram()
    pt = ggplot(turnstile_weather, aes("ENTRIESn_hourly")) + geom_histogram()
    print(pt)

def gr(df):
    #return ggplot(df, aes("Hour","ENTRIESn_hourly")) + geom_point(color = 'red')
    return( ggplot(df, aes("hour", weight = "ENTRIESn_hourly")) + geom_bar())

#print(gr(my_df))

def time_of_the_day_bargraph(datafr):

    dft = pandas.DataFrame()
    dft["sums"] = [datafr["ENTRIESn_hourly"][datafr["Hour"] == x].sum()/1000000 for x in range(0,25)]
    
    dft["Hour"] = [x for x in range(0,25)]

    pt = ggplot(dft, aes("Hour", weight = "sums")) + geom_bar(binwidth = 1)
    pt = pt + scale_x_continuous(limits=(0, 24))
    pt = pt + xlab("Hour") + ylab("Number of entries (in millions)") + ggtitle("Subway ridership depending on the time of day")

    return(pt)


def time_of_the_day_bargraph2(datafr):

    dft = pandas.DataFrame()
    dft["sums"] = [datafr["ENTRIESn_hourly"][datafr["Hour"] == x].sum() for x in range(0,25)]
    dft["counts"] = [len(df2[df2["Hour"] == x ]) for x in range(0,25)]

    dft["normalized"] = dft["sums"]/dft["counts"]

    dft["n_sums"] = dft["sums"] / max(dft["sums"])
    dft["n_normalized"] = dft["normalized"] / 2656
    
    dft["Hour"] = [x for x in range(0,25)]

    print(dft)
    pt = ggplot(dft, aes("Hour", weight = "normalized")) + geom_bar(binwidth = 1)
    pt = pt + scale_x_continuous(limits=(0, 24))
    pt = pt + xlab("Hour") + ylab("Number of entries (in millions)") + ggtitle("Subway ridership depending on the time of day")
    return (pt)

     
    
    
def rain_no_rain_graph(datafr):
    rain_labels = []
    for i in datafr["rain"]:
        if i == 0:
            rain_labels.append("no rain")
        else:
            rain_labels.append("rain")
    datafr["rl"] = rain_labels
    
    pt = ggplot(datafr, aes("ENTRIESn_hourly")) + geom_histogram(binwidth = 100)
    pt = pt + facet_grid("rl")
    pt = pt + scale_x_continuous(limits=(0, 5000))
    pt = pt + xlab("Hourly entries") + ylab("Number of entries") + ggtitle("Subway ridership on rainy vs not rainy days")
    print(pt)



    
#print (time2(my_df))

#print(my_df["Hour"].describe())
#draw_histograms(my_df)

print(time_of_the_day_bargraph(df2))

#print(my_df.shape)
#print(mann_whitney_plus_means(df))

#draw_histograms(my_df)
#print(gr(my_df))
#for i in range(0,24):
 #   print (i, ": ", my_df["ENTRIESn_hourly"][my_df["Hour"]==i].sum()/1000000)