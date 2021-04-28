---
layout: post
title:      "I made a model! How do I use it? - Beginning Model Building, Part 1"
date:       2021-04-27 14:26:09 -0400
permalink:  i_made_a_model_how_do_i_use_it_-_beginning_data_science_part_1
---

![](https://i.imgur.com/UvNR2p9.jpg)
copyright: NBC


There are SO MANY articles out there on model building. And with rare exception, each one ends right after they show a model has been produced and scored, leaving the neophyte user with a resounding "OK… but now what?" These are the gaps that technical writers don't fill, perhaps because this step is so obvious to them.

*  I made a model. How do I predict something?? And what do I predict with?
*  How do I use the model RIGHT NOW to make a totally new prediction?
*  How do I prepare the model for use later without having to re-produce the model?
	
An anecdote. My partner was working for the first time with a 3D Printer. He was using the correct software and doing all of the things that the tutorials showed for setting up an item to print. And then he was stuck on the "now what?" spot. How did he get the cleaned and prepared file into the right format to put into the 3d Printer? It turned out that all he needed to do was hit a button called "Slice", which in context, tells the program to "slice" the 3d object into the thousands of layers that are printed on the printer one-by-one. And to a veteran user of 3d printers and software, this step is very obvious. SO obvious, that no tutorial says "Hey - now click SLICE in order to take your theoretical finished object into a real finished object that you can use."

I experienced a similar confusion when it came to using my model for the first time. I took an Amazon box over to my partner and showed it to him. "Look, this is my model." I said. "That's a box," he replied. "EXACTLY," I said. "HOW DO I USE THIS BOX?" See, everyone knows you have to click the Slice button to make a 3D print file - unless you've never seen the slice button before. Someone has to tell you how to do it that very first time.

We'll talk in this three-part series about three things:

1. 	Part 1 - Using your model to make predictions on your holdout set
2. 	Part 2 - Using a model right now to make a brand new prediction
3. 	Part 3 - Preparing the model to make predictions later, without having to make the model again

# **Using your model to make predictions on your holdout set**

We don't want to go all the way back to the beginning of model building in this writeup, so we're going to start at a point that assumes you have preprocessed your data and prepared it for the type of regression that you're performing. This includes importing your data, cleaning up bad/missing values, necessary scaling, and preparing your categorical/continuous variables. Different models may benefit from different levels or types of preprocessing, but that is outside the scope of this article. You are here, with a prepared dataset, ready to make a box, and hopefully to mysteriously convert things inside. We'll call our prepared dataset "prepared_dataset"

## **Making Your Model From Your Processed Data**

If your model is built and ready to predict with, you can skip this section. But, did you divide into train/test? Did you use cross validation on your training set? If the answer is no, or even if you did these things but do not understand WHY,  you might want to slow down and follow along with me.

If you didn't divide into train/test, it's time for your first few steps back. You're going to use 70-80% of your data to make your model. The remaining percentage is your holdout data, and you will not touch it until your model is ready to test. This is because we can't use our targets to help us predict our same targets. That is very bad data science. Your bare minimum of preparation is to create a train/test split. Now mentally put the test set aside for a while. 

If your dataframe has the target variable as one of the columns, we need to pull that out. The modeler needs separate X and y inputs. X is all of your predictors, and y holds only your target dependent variable.
```
X = prepared_dataset.drop('target_variable', axis=1)
y =  prepared_dataset['target_variable']
```
Now we split up our data into train and test. Sklearn provides us with a very easy way to create this split. I'm using 20% for my test holdout, so I set test_size to .2
```
from sklearn.model_selection import train_test_split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
You'll notice I name my training sets train_val. This is to remind myself that we are doing both training and validation on this set. We DO NOT touch our test set until our model is done.

Now it's time to unpack what each of our model commands does.
```
model = LinearRegression() 
```
This is where we specify what estimator we are using. There are many to learn about, but the basic steps to fit a model are the same.

```
model.fit(X_train_val, y_train_val)
```
Right here is where the model is created - with the fit method. After this line executes, we have a completed model. We could stop right here and move on to predictions, but we want to know some scores first.

```
model.score(X_train_val, y_train_val)
```
This will give us an R^2 score for our model, if it is simply trained on all of the X_train_val set and then comparing its predictions to the y_train_val actual values

You've heard of cross validation. This is a much stronger way to determine how well your model is performing, because it creates multiple train/test partitions INSIDE your train_val set. When you test your model without any partitions, as in the code above, you're both creating and predicting on the same data. As we already mentioned earlier - it's bad data science to train and predict on the same data because it means we use our targets to predict themselves. You should always train and predict on different data chunks. Use cv as well whenever appropriate and possible.

You'll set your cv variable to the number of partitions, and then your train_val X and y sets will be divided into that many partitions.  For each partition z, the model will train on all of the other partitions and test against z, returning an R^2 score for that partition. Then you can either check the entire array of scores or, for an average of all of their performance, check the mean score.

```
cv_5 = cross_val_score(model, X_train_val, y_train_val, cv=5) # a more robust way
r2 = cv_5.mean()
```

Here's one of those "black box" notes that took me a while, and you may shake your head and say "well, that's obvious". It was not to me. Cross Validation is not ALTERING your model in any way. It is not an optimization function. All it is doing is reporting "here is how your model performs, if we make a bunch of train/test sets from your training data". It is up to YOU to re-tune the model, if possible/desirable. Then run your CV again, and repeat until you are happy with the results.

Are we happy? Are we ready to predict something? Great! It’s time to invite our test data back into the room!

## **Make Predictions on your Test Data**

Time to make a prediction on our test data!

```
test_predictions = model.predict(X_test)
```

That's it. We stuffed something into our box and it spit something back out. We pressed the Slice button! 
test_predictions is now an array of predictions. You probably want to do something useful with the predictions and check how well our model did now that it was confronted with all new data.

I like to organize my predictions vs my actuals in a dataframe, but it's not necessary.

```
predicted_prices = pd.DataFrame({"Actual": y_test, "Predicted": test_predictions)})
predicted_prices
```

![](https://i.imgur.com/SjDmZLe.png)

If you scaled your target variable, you might want to do some stuff here to unscale it to better interpret your results. For example, in the notebook I worked with, all of my prices were log transformed. So my data frame creation was actually like this:
	
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

For R^2, we can go back to our scoring function from before. This time we show it the test data to get a score, but it still uses the model built from the train data.
```
model.score(X_test, test_predictions)
```

When predicting values (like house prices), you can check Mean Absolute Error. This is the average of the summed differences of prediction minus actual, whether they are in a positive or negative direction.
Once again, be sure to use your unscaled/untransformed targets if you need to, in order to get the max information out of this error.
```
mae = round(mean_absolute_error(y_test, test_predictions), 2)
mae
```

You can also use RMSE, which is the Root Mean Squared Error. This one is a better metric to penalize BIG errors, meaning small errors in predictions will score better than big errors in predictions. This takes each difference of prediction minus actual and squares it, takes the mean of the sum of those squares, then takes the square root of the mean.
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

Here's a way to understand the difference between Mean Absolute Error and Root Mean Squared Error, and why you might want to use one over the other.

Let's say you have just two data points a and b, and you've made predictions. In example 1, P(a) is 70 over actual, and P(b) is 50 under actual. Your Mean Absolute Error here is 60. It's very straightforward - it's the average of the predictions minus actuals. Your Root Mean Squared Error is sqrt( (70^2 + 50^2)/2 )=60.8276253029822  

Now in an example, P(a) is 90 over actual and P(b) is 30 under actual. Your MAE is STILL 60, because that's still your averaged error. Your RMSE is  different though - sqrt( (90^2 + 30^2)/2 )=67.08203932499369 
Root Mean Squared Error punishes larger errors more, resulting in a higher error score on this prediction set. You have to decide on what metric suits your data. If you have a lot of outliers, your model may perform better if you minimize RMSE. 

In any case, pick the scoring metric you like. And here's a hint: You can actually change what kind of scoring you are looking at in your cross_val_score exploration, by changing the 'scoring' parameter. Like so:

```
cv_5 = cross_val_score(model, X_train_val, y_train_val, cv=5, scoring='neg_root_mean_squared_error')
```
	
The default for cross_val_score will return R^2, but if you want to explore other scoring metrics, this is how you would change that. Sklearn has several, and great documentation. Check out [Sklearn Scoring Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html).





Congratulations! You made a model on your train set, and used it to predict on your test set, as well as evaluated your errors.

## **Next up in Part 2, we'll do something even more mysterious - use our model to predict on ALL NEW data.**

