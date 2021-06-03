---
layout: post
title:      "Create an Awesome Map Viz Without a Geo-Visualization Package"
date:       2021-06-03 14:47:33 -0400
permalink:  create_an_awesome_map_viz_without_a_geo-visualization_package
---


![](https://i.imgur.com/3FdXLIQ.jpg)
Photo by <a href="https://unsplash.com/@nasa?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">NASA</a> on <a href="https://unsplash.com/s/photos/map?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  

When I work with real estate data, I find I do one particular task right when I open the data set, which is make a geo-visualization of the data. I do this without the use of any complicated packages or shapefiles. Not only is it a great way to visualize the physical space in which my housing set lives, but I can use this visualization to see other elements that might inform my target. All I need is Seaborn and a dataset with some lat/long information.

I start by loading my relevant packages and load my data set. In this example I'm using the King County housing dataset.

```
# data processing tools
import pandas as pd
import numpy as np

# Visualization tools
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

# load and look at our king county housing data
df = pd.read_csv('kc_house_data.csv')
df
```

This only works with a dataset that has lat/long, but it works amazingly. Watch as I take our dataset and produce a map in physical space of the King County area:

```
# plotting latitude and longitude as a visual scatter plot to look for location outliers

plt.figure(figsize=(25,25))

ax = sns.scatterplot(data=df, x="long", y="lat", palette="magma_r");

# save visualization to png
plt.savefig("tutorial/plain_map.png")
plt.show()
```

![](https://imgur.com/cM4gRMH.png)

With just a few lines of code I've produced a map of all of my data points. The first thing I can do is easily identify location-based outliers. For example with the King County dataset, there are just a few properties off east of longitude -121.65 that aren't anywhere near the metro area that we want to model for. I can drop those properties without worrying that I've eliminated useful data.

```
# drop the properties east of the metropolitan area
df.drop(df[df['long'] > -121.65].index, inplace=True)
```


I can show even more with this map, though. We all know that location is everything with real estate, so let's show that on the map, with just a few more lines of code.

We're going to use the median home value per zip code to rank the 70 zip codes in price from low to high. We'll do this by making a rank lookup table for our zipcodes then apply that to the data frame. 

```
# we're using the median house value for a zip code to determine the zip code's sort, so we can visualize the zip code importance

# group our dataframe by zipcode on median home price, sorted ascending. 
zipsorted = pd.DataFrame(df.groupby('zipcode')['price'].median().sort_values(ascending=True))

# rank each zip code and assign rank to new column
zipsorted['rank'] = np.divmod(np.arange(len(zipsorted)), 1)[0]+1

# function that looks up a segment that a data entry belongs to
def make_group(x, frame, column):
    '''Takes in a line, a lookup table, and a lookup column
    returns value of target column
    PARAMETERS:
    line from dataframe x
    lookup table frame
    column to return rank'''
    y = frame.loc[(frame.index == x)][column]
    z = np.array(y)
    z[0]
    return z[0]

# make a new column on our dataframe. Look up each zip entry's group, and append to the column.
df['zip_rank'] = df['zipcode'].apply(lambda x: make_group(x, zipsorted, 'rank'))
```

As a side note - don't use these ranks in your model, at least not before you split and use proper target encoding. That would be leaking your holdout data into your model. We're just using this right now as a visualization tool.

Now we'll make the same map plot as before, but this time we'll color the map using our zip ranks.

```

# plotting latitude and longitude as a visual scatter plot to look for location outliers

plt.figure(figsize=(25,25))

ax = sns.scatterplot(data=df, x="long", y="lat", hue='zip_rank', palette="magma_r");

# save visualization to png
plt.savefig("tutorial/colored_map.png")
plt.show()
```

![](https://i.imgur.com/0XctWSX.png)

Just like that, we can clearly visualize the importance of location to our housing prices, and we've made a useful visualization while we're at it, all without using any fancy packages or shapefiles!
