import pandas as pd
teams = pd.read_csv("teams.csv")
teams
import plotly.graph_objects as px
import pandas as pd
data=pd.read_csv("teams.csv")
plot=px.Figure(data=[px.Scatter(
    x=data['country'],
    y=data['athletes'],
    mode='markers',)
])
plot.update_layout(
     updatemenus=[
         dict(
             buttons=list([
                 dict(
                     args=["type","scatter"],
                    label="Scatter plot",
                     method="restyle"
                 ),
                 dict(
                     args=["type","bar"],
                      label="bar chart",
                      method="restyle"
                 ),
                 dict(
                     args=["type","pie"],
                      label="pie chart",
                       method="restyle"
                 ),
         ]),
     direction="right"
         ),
     ]
)
plot.show()
import seaborn as sns
sns.lmplot(x='athletes',y='medals',data=teams,fit_reg=True, ci=None) 
sns.lmplot(x='age', y='medals', data=teams, fit_reg=True, ci=None) 
teams.plot.hist(y="medals")
teams[teams.isnull().any(axis=1)].head(20)
teams = teams.dropna()
#he dropna() method removes the rows that contains NULL values. The dropna() method returns a new DataFrame object unless the inplace parameter is set to True , in that case the dropna() method does the removing in the original DataFrame instead.
teams.shape
train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()
# About 80% of the data the value shows(x,y) means number of rows and column taken for training the model
train.shape
# About 20% of the data number of rows and column taken for testing the data
test.shape
#accuracy matrix
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
predictors = ["athletes", "prev_medals"]
reg.fit(train[predictors], train["medals"])
predictions = reg.predict(test[predictors])
predictions.shape
test["predictions"] = predictions
test.loc[test["predictions"] < 0, "predictions"] = 0
test["predictions"] = test["predictions"].round()
from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(test["medals"], test["predictions"])
error
teams.describe()["medals"]

test["predictions"] = predictions
test[test["team"] == "BDI"]
errors = (test["medals"] - predictions).abs()
error_by_team = errors.groupby(test["team"]).mean()
medals_by_team = test["medals"].groupby(test["team"]).mean()
error_ratio =  error_by_team / medals_by_team 

import numpy as np
error_ratio = error_ratio[np.isfinite(error_ratio)]
error_ratio.plot.hist()
error_ratio.sort_values()

