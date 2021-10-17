import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('cars.xls')

scale = StandardScaler()

X = df[['Mileage', 'Cylinder', 'Doors']]
y = df['Price']

X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(X[['Mileage', 'Cylinder', 'Doors']].to_numpy())

print(X)

est = sm.OLS(y, X).fit()

est.summary()

doors_group = y.groupby(df.Doors).mean()

# more doors does not mean a higher price!
print(doors_group)