import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')
x_values = bmi_life_data[['BMI']]
y_values = bmi_life_data[['Life expectancy']]

bmi_life_model = linear_model.LinearRegression()
bmi_life_model.fit(x_values, y_values)

# plt.scatter(x_values, y_values)
# plt.plot(x_values, bmi_life_model.predict(x_values))
# plt.show()
laos_life_exp = bmi_life_model.predict(21.07931)
print(laos_life_exp)
