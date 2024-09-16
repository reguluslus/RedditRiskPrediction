from interpret.glassbox import ExplainableBoostingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv('../data/datasets/training_data.csv', index_col=False, skiprows=1)
X=df.iloc[:,:-1]
y=df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y,random_state=42)


ebm= ExplainableBoostingClassifier(inner_bags= 10, max_bins= 32, max_leaves=2, outer_bags= 100)
ebm.fit(X_train, y_train)


ebm_global = ebm.explain_global(name='EBM')

for i in range(len(X.columns)):
    fig = ebm_global.visualize(key=i)
    fig.write_image("f"+str(i)+".png")