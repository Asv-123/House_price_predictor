import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt



def visualise(y_pred,y_test):
  plt.figure(figsize = (6,6))
  sns.scatterplot(x = y_test.values.flatten(), y = y_pred, color = 'blue')
  plt.xlabel('Actual Values')
  plt.ylabel('Predicted Values')
  plt.title('Linear Regression for House price prediction')
  plt.show()




data = pd.read_csv('/content/BostonHousing.csv')

x = data[['crim','rm','age','ptratio','lstat']]
y = data[['medv']]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# print(x_train,x_test,y_train,y_test)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
y_pred = y_pred.flatten()

visualise(y_pred,y_test)

c_user = float(input("Enter the Crime rate"))
r_user = float(input("Enter the no of rooms"))
a_user = float(input("Enter the percentage of old buildings"))
p_user = float(input("Enter the pupil to teacher ratio"))
l_user = float(input("Enter the percent of lower status"))

dictx = {'crim':c_user,'rm':r_user,'age':a_user,'ptratio':p_user,'lstat':l_user}

user_data = pd.DataFrame([dictx])
price = model.predict(user_data)
print(f"the price of the house would be {price[0][0]}")
