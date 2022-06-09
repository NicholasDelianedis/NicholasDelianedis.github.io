---
layout: post
title: Group Submission
---

## §1. Introduction

We wanted to find out whether or not quarterly earnings conferences give any extra insight into how a company is doing financially compared to the raw data.  To this end, we regressed daily price changes with different aspects of earnings reports with linear regression and neural network models, and did a sentiment analysis on earnings conferences to see if either one has any relation, or if one has a stronger relation with future changes in value than the other.  Here is a link to our GitHub page.

![GS-flowchart.png]({{ site.baseurl }}/images/GS-flowchart.png)

## §2.1 Regression

To start with the numerical analysis, we created a linear regression model, which took in the numerical values from the dataset, and regressed them onto our variable 'DeltaPrice', which we created by subtracting the close date from the open date of the next day for each datapoint.  We then used sklearn's LinearRegression model to regress the data, and put all of this, as well as a train-test-split testing mechanism into a function that takes in a subset of our main dataframe to perform the linear regression.  With the whole dataset, the linear regression performed at around 31% accuracy, which we considered to be good, since it is predicting the exact change in price, for which there are many values.  Below is the code for the function:

```python
def lr(df = newdata):
  mlinx = []
  mliny = []
  regx = df.drop("DeltaPrice", axis = 1)
  for i in df["DeltaPrice"]:
    mliny.append(round(i, 2))
    mlinx.append([])
  for i in regx.columns:
    k = 0
    for j in regx[i]:
      mlinx[k].append(j)
      k += 1
  
  X2 = sm.add_constant(mlinx)
  est = sm.OLS(mliny, X2)
  est2 = est.fit()
  print(est2.summary())
  j = 1
  for i in regx.columns:
    print(str(j) + ": " + i)
    j += 1
  X_train, X_test, y_train, y_test = train_test_split(mlinx, mliny, test_size = 0.2, random_state = 42)
  LR = LinearRegression().fit(X_train, y_train)
  print("Test score:")
  print(LR.score(X_train, y_train))
  return X_train, y_train
```

For the neural networks, we used a similar format as for the linear regression model, but we changed it so that the user can input which sector they would like to run the regression on, since the model is too big to run on the whole data, and we got better results from running it on just one sector than on the whole dataset, and whether they would like to use the model with or without the LSTM, or long short term memory, layer.  We found that the model works moderately well without the LSTM, but produces meaningless results with the LSTM layer, so the default setting is without the LSTM layer.  Below is the code for the function, and a screenshot of some of the last epochs of the model run without LSTM on the sector 'Healthcare', which have a loss of around 20.

```python
def nn(sector="Technology", rnn = False):
  rnndf = regdata[regdata["Sector"] == sector]
  rnndf = rnndf.drop(["Sector", "Industry", "Date", "NextDay"], axis = 1)
  rnndf = rnndf[rnndf["Close"] != 0]
  rnndf["PctChg"] = rnndf.apply(lambda row: round(1000 * (row["DeltaPrice"] / row["Close"])) / 10, axis=1)
  rnndf = rnndf[rnndf["PctChg"] < 70]
  rnndf = rnndf[rnndf["PctChg"] > -70]
  rnnx = rnndf.drop(["DeltaPrice", "PctChg"], axis = 1)
  cmp = rnndf["SimFinId"]
  cmp = cmp.drop_duplicates()
  ryl = []
  rny = rnndf.sort_values("NewDate", axis = 0)
  for i in rny["PctChg"]:
    ryl.append(round(i, 2))
  rxl = []
  rnx = rnnx.sort_values("NewDate")
  for i in rnx["NewDate"]:
    rxl.append([])
  for i in rnx.columns:
    k = 0
    for j in rnx[i]:
      rxl[k].append(j)
      k += 1
  rxa = np.array(rxl)
  rya = np.array(ryl)
  if rnn:
    model.fit(rxa, rya, epochs=50, verbose=1)
  else:
    model1.fit(rxa, rya, epochs=500, verbose=1)
```

![GS-nn.png]({{ site.baseurl }}/images/GS-nn.png)