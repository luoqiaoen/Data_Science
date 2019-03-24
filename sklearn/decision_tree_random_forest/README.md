

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
df = pd.read_csv('kyphosis.csv')
```

## Let's have a look


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Kyphosis</th>
      <th>Age</th>
      <th>Number</th>
      <th>Start</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>absent</td>
      <td>71</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>absent</td>
      <td>158</td>
      <td>3</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>present</td>
      <td>128</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>absent</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>absent</td>
      <td>1</td>
      <td>4</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(df,hue='Kyphosis',palette='Set2')
```

    /Users/luoqiaoen/anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval





    <seaborn.axisgrid.PairGrid at 0x1a16ec5b70>




![png](Decision%20Trees%20and%20Random%20Forests%20in%20Python_files/Decision%20Trees%20and%20Random%20Forests%20in%20Python_4_2.png)


# Train Test Split


```python
from sklearn.model_selection import train_test_split
X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
```

## Decision Trees


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
dtree = DecisionTreeClassifier()
```


```python
dtree.fit(X_train,y_train)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')



## Evaluate Model


```python
predictions = dtree.predict(X_test)
```


```python
from sklearn.metrics import classification_report,confusion_matrix
```


```python
print(classification_report(y_test,predictions))
```

                  precision    recall  f1-score   support
    
          absent       0.76      0.89      0.82        18
         present       0.50      0.29      0.36         7
    
       micro avg       0.72      0.72      0.72        25
       macro avg       0.63      0.59      0.59        25
    weighted avg       0.69      0.72      0.69        25
    



```python
print(confusion_matrix(y_test,predictions))
```

    [[16  2]
     [ 5  2]]



```python

```
