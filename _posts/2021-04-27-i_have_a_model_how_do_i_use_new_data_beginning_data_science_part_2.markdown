---
layout: post
title:      "I have a model! How do I use new data? Beginning Model Building, Part 2"
date:       2021-04-27 20:32:43 -0400
permalink:  i_have_a_model_how_do_i_use_new_data_beginning_data_science_part_2
---

![](https://i.imgur.com/SQ6nnoY.jpg)

Welcome to Part 2 of our beginner's series on using our model. [Part 1](https://threnjen.github.io/i_made_a_model_how_do_i_use_it_-_beginning_data_science_part_1) back there included elements on fitting the model, train/test split, cross validation, and predicting on our test/holdout data. If you're comfortable with all of those things, forge onward. Otherwise, pop back to that entry first.

## **We have a box! And we're very excited about it!**

But no one teaches us how to use our model to make NEW predictions with entirely new data. The model just sits there like a box on our floor. We trained it properly using a train/test split and cross validation, and then tested it against our holdout data. We're ready to use it to do something new and exciting, but we have no idea how to do that in a user-friendly way.

## **Our first obstacle - How do we persist our feature standardization/transformation?**

If we scaled our features when building our model, it should seem obvious that we must scale our features to use the model. How does this work persistently? While we are told TO scale our features, we are never shown HOW that works across feature sets. Because imagine this - you have your initial data which you have standardized using the sklearn StandardScaler black box function. We want to make new predictions with all new data. How do we scale a single point of data, when scaling by its nature involves accessing all of the data to determine a distribution?

**We can do this in a couple of ways:**

* Using StandardScaler/MinMaxScaler by understanding how their use creates an object that we can apply to other data, or
* Write our own standardization function and save our function variables for later application
* Some other way that is outside the scope of this article

We'll use the first two methods and either of these methods is acceptable. Our own standardization function is less opaque and "black box" than an sklearn scaler, but will involve more code writing on our part.

To use StandardScaler or MinMaxScaler, we need only a few lines.
```
scaler = StandardScaler()
scaler.fit(X_train_val)
```

Just like when using the fit method made our model spring into existence, we now have a scaler object. We can apply this object to other sets of data or even single data points and it will be scaled using the scale stored in the object. We would now apply this scaler to our train and test sets:
```
X_train_val_scaled = scaler.transform(x_train_val)
X_test_scaled = scaler.transform(x_test)
```


It's worth noting - we only scale continuous variables. You don't want to inadvertently apply this to one-hot-encoded flag variables. So you might need to take a few extra steps to separate your data, scale the continuous features, and put it back together.

Our other method of standardization is to write a custom function ourselves that is based on our data. In this example I'm using my data frame df_continuous which has only my continuous variables.

```
standardization_coeffs = {}

for item in df_continuous:
    standardization_coeffs[item+'_mean'] = df_continuous[item].mean()
    standardization_coeffs[item+'_std'] = df_continuous[item].std()

print(standardization_coeffs)

{'sqft_living_mean': 7.487264646488237, 'sqft_living_std': 0.3868656778159048, 'sqft_lot_mean': 8.751422942498527, 'sqft_lot_std': 0.6052323520926731, 'bedrooms_mean': 1.161988952164995, 'bedrooms_std': 0.28071365873779, 'condition_mean': 1.2099855036820262, 'condition_std': 0.18008256313016827, 'grade_mean': 2.003972468584745, 'grade_std': 0.13169146539316434}
```

I've made a nice dictionary of standardization coefficients that I can call on to standardize this kind of data. And although we'll talk more later about saving our model assets for later use, I'll mention that I export this to a csv.

```
# save standardization coefficients to file so I can use these later
with open('standardization_coeffs.csv','w', newline="") as f:
    w = csv.writer(f)
    w.writerows(standardization_coeffs.items())
```

And I write a function that I can use to standardize any needed column later on

```
def standardize_continuous(key, value):
    """Standardize a value according to saved mean and standard deviation saved values"""
    transformed = (np.log(value) - standardization_coeffs[key+'_mean']) / standardization_coeffs[key+'_std']
    return transformed

```

Using either a black box scaler or your own custom scaler is a valid choice. It just depends if you want brevity versus transparency.


## **Our second obstacle - How do we feed new data into our model?**

Our model expects to see new data with the same predictor variables with which the model was created. If predictors are missing, the prediction attempt will throw an error. This can seem overwhelming if you have high cardinality categoricals that you have one-hot encoded. I'll help you set up a dataframe which can be reused whenever you want to predict new data with your model.

First we need to set up a dataframe that has all possible predictors included and with all predictors set to 0. This will ensure that our default one-hot-encoding flags are all set to 0, so that we only have to change the relevant entries to a 1.

We define our columns using the predictor set that we used to train our final model. In my case that was X.

```
test_frame = pd.DataFrame(0, index=range(1), columns=X.columns)
test_frame
```
![](https://i.imgur.com/k8erfwr.png)

This next part is going to require you to write good documentation for yourself to use later. You're going to set up a section in your code to enter your new predictors. Here is where you'll need to make decisions about what kind of usability you want to integrate. Do you want to write yourself a fancy input GUI? How much do you want to think about the format of the values that you need to enter later on? My suggestion is to follow these guidelines to set up your data entry space:
	
* Make it only as complicated as needed
* Avoid unnecessary GUI elements unless you need them or someone else will use the tool
* Write comments on all of the inputs to remind yourself what to enter. More comments is better. Better to over-explain to future you than under-explain.
* When possible, use the most intuitive input, or the input type closest to the data you will have, or have a conversion function already prepared so that you don't have to spend extra time pre-processing your new data
* Keep your data entry area as neat and tidy as possible, to avoid confusing yourself when you approach it later

My example data has several different categories of data, although at this juncture it is all numerical. I have plain old numerical continuous data. I have numerical data that is actually categorical and will become a one-hot encoded flag. I have dichotomous one-hot encoded data which is either yes/no. And I have numerical data which will be sorted into a bin, which is the most complicated type I'll be presenting.

All of this data will need to be prepped in a manner similar to preprocessing for your model, with the added step of adding it into the correct place in your predictor data frame.

* Continuous numerical data. This data will need to be log transformed and then standardized. I'll leave myself a comment even though these should be straightforward

```
 # these should all be numbers
sqft_living = 1960
sqft_lot = 5000
bedrooms = 4
```

* Ordinal data, where higher is better. I need to make sure I leave myself excellent documentation on these entries so that I know what the numbers mean when I come back to this model. This data will also be transformed and standardized.

```
 # The description here is: Overall property condition
 # The choices here are: Poor, Okay, Average, Good, Excellent range 1-5
 # provide examples to indicate that this variable indicates property repair/maintenance level, not quality of materials
condition = 3

 # Give examples within each category so they can make a best guess. Ex. Low Quality - Linoleum >20 yrs old, Laminate counters
 # ex. continued - Very High Quality - crown moulding, solid slab granite. 
 # provide examples to allow proper selection of grade range 1-10
grade = 5
```

* Dichotomous categoricals, meaning they are only a yes or a no. I could take an extra step and allow myself to write 'yes' or 'no' for these spots and then interpret, but I can also leave myself a reminder that 1 is yes and 0 is no.

```
 # 1 for yes or 0 for no
waterfront = 0
renovated = 1
basement = 0
```

* High-cardinality categoricals that correspond directly to a one-hot encoding. I will write functions to find and change the relevant encoding flags to 1 for these entries. If there is any uncertainty on format, I should leave myself a comment.

```
zipcode = 98136
month = 12 # Month should be a number for the month 1=January  to 12=December
```

* Binned categoricals. This is a one-hot encoded categorical that I binned into intervals. For this entry I will need to find the interval that the entry belongs to, and then change the relevant bin entry to a 1.
```
year_built = 1965
```

We're building some dictionaries from this data to process quicker.

```
continuous = {'sqft_living':sqft_living, 'sqft_lot':sqft_lot, 'bedrooms':bedrooms, 'condition':condition, 'grade':grade}
dichotomous = {'waterfront':waterfront, 'renovated':renovated, 'basement':basement}
high_card_cat = {'zipcode': zipcode, 'month':month}
binned = {'year_built':year_built}
```


We're going to save all of our new parameters to a dictionary test_parameters, then apply that to our test_frame dataframe. 

### **Transform and standardize our continuous variables** 

We need to transform and standardize our continuous variables in the same manner that we did it for the continuous variables in our train set. So if we used our StandardScaler(), or MinMaxScaler(), time to pull that from the back pocket. Or if we wrote a more transparent function for ourselves, like my standardize_continuous function, it's time to lock and load.

If we used a black box scaler, this is as easy as:

```
For item in continuous:
	value = np.log(continuous[item]) # log transform our value first, if applicable
	test_parameters[item] = scaler.transform(value) # apply our saved scaler
```

For our custom scaled coefficients we use our standardization function that we wrote earlier.

```
for item in continuous:
    test_parameters[item] = standardize_continuous(item, continuous[item])
```

Either way, our continuous variables are scaled and ready.

### **Flag our high cardinality categoricals**

One thing you might recall from preparing categoricals is that one categorical is always thrown out, or a feature may have been thrown out during feature selection. So the column that we want to flag might not even be present in our data frame. We will account for this possibility.

We need to check out our data frame and see how the categoricals were named. In our example, the default categorical name is "variable_data", for example our column for zipcode variable with zip 98136 would be called "zipcode_98136". The naming convention should be consistent for your categoricals, and allow you to write conversion code that captures all of them at once.

This code first checks if the column we want is in our data frame. If the column is there, it changes it to a 1.

```
for item in high_card_cat:
    if item+'_'+str(high_card_cat[item]) in test_frame.columns:
        test_parameters[item+'_'+str(high_card_cat[item])] = 1
```

### **Flag our dichotomous categoricals**

Dichotomous categoricals get named with the same conventions as the high cardinality ones, but since there is only a yes or no option, only a "yes" column should exist for them, in whatever format. Check the format, then write code that flags that column if you entered a yes.

```
for item in dichotomous:
    if dichotomous[item]:
        test_parameters[item+'_1.0']=1
```

### **Identify Binned Categoricals**

Binned categoricals are the most complicated to sort and find. They will have a naming convention, but it will be more complicated. Make sure that when you create these bins using either pandas cut or qcut (outside the scope of this article), you did retbins=True and saved your bins to a variable. You'll need that variable now in order to figure out where your new data fits into the bins.

In order to find the proper data frame column to flag you'll need your lower and upper bin bound. I wrote a simple function that takes in the variable and the bins and produces the bin bounds for your variable.

```
def age_block_finder(year, bins):
    for i in range(len(bins)):
        if year > bins[i] and year < bins[i+1]:
            lower_year, upper_year = bins[i], bins[i+1]
        else: continue
    return lower_year, upper_year

lower_year, upper_year = age_block_finder(year_built, year_bins)
```

I then use those bounds to first find out if this predictor is in our data frame at all, and if it is, to flag it as a 1. Once again, I'll need to check my data frame for the formatting of the column label. Remember that the column label itself is  a string, so we need to find it in a very literal manner.

```
if 'year_block_('+str(lower_year)+', '+str(upper_year)+']' in test_frame.columns:
    test_parameters['year_block_('+str(lower_year)+', '+str(upper_year)+']'] = 1
```

### **Prepare our new prediction data frame**

We now have a dictionary of the test parameters that we'll plug into our test_frame data frame.

```
Test_parameters

 {'sqft_living': 0.24151820927569645,
 'sqft_lot': -0.38700798176503226,
 'bedrooms': -0.8752627905446619,
 'condition': -0.618456407316976,
 'grade': -2.9959007212256563,
 'zipcode_98136': 1,
 'month_12': 1,
 'lat_block_(47.512, 47.523]': 1,
 'renovated_1.0': 1}
 ```

Now we'll apply it to our test frame that we prepared earlier.

```
for item in test_parameters:
    value = test_parameters[item]
    test_frame[item] = value

test_frame
```
![](https://i.imgur.com/ta7Cip3.png)

## Pulling all of the above together into ONE function

We can pull all of the above elements together into a single function. We pass it our dictionaries of variables, our year and lat bins, and the column names for our predictors. 


```
def predict_from_one(continuous, dichotomous, high_card_cat, binned, year_bins, lat_bins, columns):

    # create an empty dictionary to store our parameters
    test_parameters = {}
    
    # create our predictor data frame full of 0s
    test_frame = pd.DataFrame(0, index=range(1), columns=columns)

    # standardize and store our continuous variables
    for item in continuous:
        test_parameters[item] = standardize_continuous(item, continuous[item])
    
    # This code first checks if the column we want is in our data frame. If the column is there, it changes it to a 1.
    for item in high_card_cat:
        if item+'_'+str(high_card_cat[item]) in test_frame.columns:
            test_parameters[item+'_'+str(high_card_cat[item])] = 1
        
    # This code first checks if the column we want is in our data frame. If the column is there, it changes it to a 1.
    for item in dichotomous:
        if dichotomous[item]:
            test_parameters[item+'_1.0']=1

    # for our categoricals, not all are used in our model. For each categorical that we would create with our entered data,
    # we check first and see if it's in our model at all. If so we flag it as a 1, otherwise it is ignored.

    # function to find lower and upper bin bounds for our age blocks
    def age_block_finder(year, bins):
        for i in range(len(bins)):
            if year > bins[i] and year < bins[i+1]:
                lower_year, upper_year = bins[i], bins[i+1]
            else: continue
        return lower_year, upper_year

    # function to find lower and upper bin bounds for our latitude blocks
    def lat_block_finder(lat, bins):
        for i in range(len(bins)):
            if lat_round > bins[i] and lat_round < bins[i+1]:
                lower_lat, upper_lat = round(bins[i], 3), round(bins[i+1], 3)
            else: continue
        return lower_lat, upper_lat


    lower_year, upper_year = age_block_finder(year_built, year_bins) # lower and upper bounds for our age block
    lower_lat, upper_lat = lat_block_finder(lat_round, lat_bins) # lower and upper bounds for our latitude block

    # Find the correct bin for our year built
    if 'year_block_('+str(lower_year)+', '+str(upper_year)+']' in test_frame.columns:
        test_parameters['year_block_('+str(lower_year)+', '+str(upper_year)+']'] = 1

    # find the correct bin for our latitude
    if 'lat_block_('+str(lower_lat)+', '+str(upper_lat)+']' in test_frame.columns:
        test_parameters['lat_block_('+str(lower_lat)+', '+str(upper_lat)+']'] = 1
        
    # enter all of our predictors into our predictor frame
    for item in test_parameters:
        value = test_parameters[item]
        test_frame[item] = value
    
		# send the predictor frame to the model and get a result
    predicted_price = int(np.exp(final_model.predict(test_frame)))    
    
    return predicted_price
		```

## Make a prediction with new data!

The moment has come! We're going to put something TOTALLY NEW in our box and it's going to give us something back!
```
predicted_price = predict_from_one(continuous, dichotomous, high_card_cat, binned, year_bins, lat_bins, columns)
```

![](https://i.imgur.com/GTXLco6.jpg)

Predict is such a magic button. 

Make sure that if you need to, you reverse transform your prediction. In my example my target price was log transformed, so I actually wrote this in order to get a usable number back out:
```
predicted_price = int(np.exp(final_model.predict(test_frame)))
predicted_price

384005
```

And, since we were kind to ourselves and set up functions to properly place our respective features, we can come back to this notebook any time and make new predictions with a minimum of effort, as well as use the function on a larger csv with multiple new entries.

