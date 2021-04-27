---
layout: post
title:      "I made a model! How do I use it? - Beginning Data Science, Part 1"
date:       2021-04-27 14:26:09 -0400
permalink:  i_made_a_model_how_do_i_use_it_-_beginning_data_science_part_1
---


There are SO MANY articles out there on model building, for every type of model. And each and every one, with rare exceptions, stops right at the point after they show a model has been produced and scored - and then the article ends. Leaving the neophyte user with a resounding "OK… now what?"

*  I made a model. How do I predict something?? And what do I predict with?
*  How do I use the model RIGHT NOW to make a totally new prediction?
*  How do I prepare the model for use later without having to re-produce the model?
	
These are the gaps that technical writers aren't filling, perhaps because this step is something that they have done so many times that it is just obvious to them.

An anecdote. My partner, who happens to be a data scientist himself, was working for the first time with a 3D Printer. He was using the correct software and doing all of the things that the tutorials showed him for setting up an item to print. And then he was stuck on the "now what?" spot. How did he get the cleaned and prepared file into the right format to put into the 3d Printer? It turned out that all he needed to do was hit a button called "Slice", which in context, tells the program to "slice" the 3d object into the thousands of layers that are printed on the printer one-by-one. And to a veteran user of 3d printers and software, this step is very obvious. SO obvious, that no tutorial says "Hey - now click SLICE in order to take your theoretical finished object into a real finished object that you can use."

I experienced a similar confusion when it came to using my model. I took an Amazon box over to my partner and showed it to him. "Look, this is my model." I said. "That's a box," he replied. "EXACTLY," I said. "HOW DO I USE THIS BOX?"

And this is where so many articles and lessons fail to bridge the gap between the theoretical things that we are learning as data scientists, and the practical application of that science. Here, I've made a model! That's great -  but HOW DO I USE IT TO DO SOMETHING? The data scientists who already have this training know how to put out that predict line and make a new prediction. For the rest of us, this isn't quite so obvious, and it's mysteriously under-explained. Everyone knows you have to click the Slice button - unless you've never seen the slice button before. Someone has to tell you how to do it that very first time.

We'll talk in this three-part series about three things:

1. 	Part 1 - Using your model to make predictions on your holdout set
1. 	Part 2 - Using a model right now to make a brand new prediction
1. 	Part 3 - Preparing the model to make predictions later, without having to make the model again

# **Using your model to make predictions on your holdout set**

We don't want to go all the way back to the beginning of model building in this writeup, so we're going to start at a point that assumes you have preprocessed your data and prepared it for the type of regression that you're performing. This involves importing your data, cleaning up bad/missing values, necessary scaling, and preparing your categorical/continuous variables. Different models may benefit from different levels or types of preprocessing, but that is outside the scope of this article. You are here, with a prepared dataset, ready to make a box, and hopefully to mysteriously convert things inside. We'll call our prepared dataset "prepared_dataset"

## **Making Your Model From Your Processed Data**

If your model object is ready to predict with, you can skip this section. But, did you divide into train/test? Did you use cross validation on your training set? If the answer is no, or even if you are not sure WHY you did these things even if you DID do them, you might want to slow down and follow along with me.

Did you divide your data into train and test? If not, it's time for your first few steps back. You're going to use 70-80% of your data to make your model. The remaining percentage is your holdout data, and you will not touch it until your model is ready to test. This is because we can't use our targets to help us predict our same targets. That is very bad data science. Your BARE MINIMUM of preparation is to create a train/test split. Now look away from the test set for a while. We won't be needing it again for a time.

If your dataframe has your target variable as one of the columns, we first need to pull that out. In order to fit, the model needs separate X and y inputs. X is all of your predictors, and y holds only your target dependent variable.
```
X = prepared_dataset.drop('target_variable', axis=1)
y =  prepared_dataset['target_variable']
```
Now we split up our data into train and test. Sklearn provides us with a very easy way to create this split. I'm using 20% for my test holdout, so I set test_size to .2
```
from sklearn.model_selection import train_test_split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=randomstate)
```
You'll notice I name my training sets train_val. This is to remind myself that we are doing both training and validation on this set. We DO NOT touch our test set until our model is done.

Now it's time to unpack what each of our model commands does.
```
model = LinearRegression() 
```
This is where we specify what estimator we are using. There are lots!

```
model.fit(X_train_val, y_train_val)
```
Right here is where the model is created - with the fit method. After this line executes, we have a completed model. We could stop right here and move on to predictions, but we want to know some scores first.

```
model.score(X_train_val, y_train_val)
```
This will give us an R^2 score for our model, if it is simply trained on all of the X_train_val set and then comparing its predictions on the y_train_val actual values

You've certainly heard of cross validation. This is a much stronger way to determine how well your model is performing, because it creates multiple train/test partitions INSIDE your train_val set. When you test your model without any partitions, you're both creating and predicting on the same data. As we already mentioned earlier - it's bad data science to train and predict on the same data because it means we use our targets to predict themselves. You should always be using some form of cv to check your model, because then it is always training and predicting on different data.

Whatever you set your cv variable to, your train_val X and y sets will be divided into that many partitions, and for each partition z, the model will train on the other partitions and test against z, returning an R^2 score for that partition. Then you can either check the entire array of scores or, for an average of all of their performance, check the mean score.

```
cv_5 = cross_val_score(model, X_train_val, y_train_val, cv=5) # a more robust way
r2 = cv_5.mean()
r2
```

Here's one of those "black box" notes that took me a while, and you may shake your head and say "well, that's obvious". But it was not to me. Cross Validation is not ALTERING your model in any way. It is not an optimization function. All it is doing is saying "here is how your model performs, if we make a bunch of train/test sets from your training data". It is up to YOU to re-tune the model, if possible/desirable. Then run your CV again, and repeat until you are happy with the results.

Are we happy? Are we ready to predict something? Great! It’s time to invite our test data back into the room!

## **Make Predictions on your Test Data**

We invite our test data back into the game now that our model is done. Time to make a prediction!

```
test_predictions = model.predict(X_test)
```

That's it. Test_predictions is now an array of predictions. You probably want to do something useful with them and check how well our model did now that it was confronted with all new data.

I like to organize my predictions vs my actual in a dataframe, but that's certainly optional.

```
predicted_prices = pd.DataFrame({"Actual": y_test, "Predicted": test_predictions)})
predicted_prices
```

![](https://i.imgur.com/SjDmZLe.png)

If you scaled your target variable, you might want to do some stuff here to unscale it. For example, in the notebook I worked with for this example, all of my prices were log transformed. So my data frame creation was actually like this:
	
```
# get our actual prices reverse log transformed
test_actual = np.exp(y_test)
	
# reverse log transform our predicted values
test_predictions_unscaled = np.exp(test_predictions)

# compare our predicted values to the actual values
predicted_prices_basiclr = pd.DataFrame({"Actual": test_actual, "Predicted": test_predictions_unscaled})
predicted_prices_basiclr
```



Now that you have predictions on your test data, you probably want to determine how well your model performed.

For R^2, we can go back to our scoring function from before. But this time we show it the test data, and it still uses the model it built from the train data.
```
model.score(X_test, y_test)
```

When predicting values (like house prices), I like to use Mean Absolute Error. This is the average of all of the prediction errors, whether they are positive or negative errors.
Once again, be sure to use your unscaled/untransformed targets if you need to, in order to get the max information out of this error.
```
mae = round(mean_absolute_error(y_test, test_predictions), 2)
mae
```

You can also use RMSE, which is the Root Mean Squared Error. This one is a better metric to penalize BIG errors, meaning small deviations are better than big deviations. 
```
rmse = round(np.sqrt(mean_squared_error(y_test, test_predictions)), 2)
rmse
```

We also might want to visualize our model's predictions versus actuals. Keep in mind the example is for my data.
```
plt.figure(figsize=(10,5))
sns.distplot(y_test, hist=True, kde=False)
sns.distplot(test_predictions, hist=True, kde=False)
plt.legend(labels=['Actual Values of Price', 'Predicted Values of Price'])
plt.xlim(0,);
```
![](https://i.imgur.com/SSdzzMA.png)

### Understanding MAE vs RMSE

Here's a way to understand the difference between MAE and RMSE, and why you might want to use one over the other.

Let's say you have just two data points a and b, and you've made predictions. In example 1, P(a) is 70 over actual, and P(b) is 50 under actual. Your Mean Absolute Error here is 60. Very straightforward. Your root mean squared error is sqrt( (70^2 + 50^2)/2 )=60.8276253029822  

Now in an example, P(a) is 90 over actual and P(b) is 30 under actual. Your MAE is STILL 60, because that's still your averaged error. Your RMSE is  different though - sqrt( (90^2 + 30^2)/2 )=-67.08203932499369 
Root Mean Squared Error punishes larger errors more, resulting in a higher error score. You have to decide on what metric suits your data. If you have a lot of outliers, your model may perform better if you minimize RMSE. 

In any case, pick the scoring metric you like. And here's a hint: You can actually change what kind of scoring you are looking at in your cross_val_score exploration, by changing the 'scoring' parameter. Like so:

```
cv_5 = cross_val_score(model, X_train_val, y_train_val, cv=5, scoring='neg_root_mean_squared_error')
```
	
The default for cross_val_score will return R^2, but if you want to explore other scoring metrics, this is how you would change that. Sklearn has several, and great documentation. Check out https://scikit-learn.org/stable/modules/model_evaluation.html



Congratulations! You made a model on your train set, and used it to predict on your test set, as well as evaluated your errors.

**### Next up in Part 2, we'll do something even more mysterious - use our model to predict on ALL NEW data.**

