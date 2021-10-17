import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

input_file = "PastHires.csv"
df = pd.read_csv(input_file, header=0)

# scikit-learn needs everything to be numerical for decision trees to work. So, we'll map Y,N to 1,0
d = {'Y': 1, 'N': 0}
df['Hired'] = df['Hired'].map(d)
df['Employed?'] = df['Employed?'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)
d = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(d)

# Next we need to separate the features from the target column that we're trying to bulid a decision tree for
features = list(df.columns[:6])

y = df["Hired"]
X = df[features]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, y)

# Predict employment of an employed 10-year veteran
print(clf.predict([[10, 1, 4, 0, 0, 0]]))
# and an unemployed 10-year veteran
print(clf.predict([[10, 0, 4, 0, 0, 0]]))