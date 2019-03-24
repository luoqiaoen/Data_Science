

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
```




    dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])




```python
print(cancer['DESCR'])
```

    .. _breast_cancer_dataset:
    
    Breast cancer wisconsin (diagnostic) dataset
    --------------------------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 569
    
        :Number of Attributes: 30 numeric, predictive attributes and the class
    
        :Attribute Information:
            - radius (mean of distances from center to points on the perimeter)
            - texture (standard deviation of gray-scale values)
            - perimeter
            - area
            - smoothness (local variation in radius lengths)
            - compactness (perimeter^2 / area - 1.0)
            - concavity (severity of concave portions of the contour)
            - concave points (number of concave portions of the contour)
            - symmetry 
            - fractal dimension ("coastline approximation" - 1)
    
            The mean, standard error, and "worst" or largest (mean of the three
            largest values) of these features were computed for each image,
            resulting in 30 features.  For instance, field 3 is Mean Radius, field
            13 is Radius SE, field 23 is Worst Radius.
    
            - class:
                    - WDBC-Malignant
                    - WDBC-Benign
    
        :Summary Statistics:
    
        ===================================== ====== ======
                                               Min    Max
        ===================================== ====== ======
        radius (mean):                        6.981  28.11
        texture (mean):                       9.71   39.28
        perimeter (mean):                     43.79  188.5
        area (mean):                          143.5  2501.0
        smoothness (mean):                    0.053  0.163
        compactness (mean):                   0.019  0.345
        concavity (mean):                     0.0    0.427
        concave points (mean):                0.0    0.201
        symmetry (mean):                      0.106  0.304
        fractal dimension (mean):             0.05   0.097
        radius (standard error):              0.112  2.873
        texture (standard error):             0.36   4.885
        perimeter (standard error):           0.757  21.98
        area (standard error):                6.802  542.2
        smoothness (standard error):          0.002  0.031
        compactness (standard error):         0.002  0.135
        concavity (standard error):           0.0    0.396
        concave points (standard error):      0.0    0.053
        symmetry (standard error):            0.008  0.079
        fractal dimension (standard error):   0.001  0.03
        radius (worst):                       7.93   36.04
        texture (worst):                      12.02  49.54
        perimeter (worst):                    50.41  251.2
        area (worst):                         185.2  4254.0
        smoothness (worst):                   0.071  0.223
        compactness (worst):                  0.027  1.058
        concavity (worst):                    0.0    1.252
        concave points (worst):               0.0    0.291
        symmetry (worst):                     0.156  0.664
        fractal dimension (worst):            0.055  0.208
        ===================================== ====== ======
    
        :Missing Attribute Values: None
    
        :Class Distribution: 212 - Malignant, 357 - Benign
    
        :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian
    
        :Donor: Nick Street
    
        :Date: November, 1995
    
    This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.
    https://goo.gl/U2Uwz2
    
    Features are computed from a digitized image of a fine needle
    aspirate (FNA) of a breast mass.  They describe
    characteristics of the cell nuclei present in the image.
    
    Separating plane described above was obtained using
    Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree
    Construction Via Linear Programming." Proceedings of the 4th
    Midwest Artificial Intelligence and Cognitive Science Society,
    pp. 97-101, 1992], a classification method which uses linear
    programming to construct a decision tree.  Relevant features
    were selected using an exhaustive search in the space of 1-4
    features and 1-3 separating planes.
    
    The actual linear program used to obtain the separating plane
    in the 3-dimensional space is that described in:
    [K. P. Bennett and O. L. Mangasarian: "Robust Linear
    Programming Discrimination of Two Linearly Inseparable Sets",
    Optimization Methods and Software 1, 1992, 23-34].
    
    This database is also available through the UW CS ftp server:
    
    ftp ftp.cs.wisc.edu
    cd math-prog/cpo-dataset/machine-learn/WDBC/
    
    .. topic:: References
    
       - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction 
         for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on 
         Electronic Imaging: Science and Technology, volume 1905, pages 861-870,
         San Jose, CA, 1993.
       - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and 
         prognosis via linear programming. Operations Research, 43(4), pages 570-577, 
         July-August 1995.
       - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
         to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) 
         163-171.



```python
cancer['feature_names']
```




    array(['mean radius', 'mean texture', 'mean perimeter', 'mean area',
           'mean smoothness', 'mean compactness', 'mean concavity',
           'mean concave points', 'mean symmetry', 'mean fractal dimension',
           'radius error', 'texture error', 'perimeter error', 'area error',
           'smoothness error', 'compactness error', 'concavity error',
           'concave points error', 'symmetry error',
           'fractal dimension error', 'worst radius', 'worst texture',
           'worst perimeter', 'worst area', 'worst smoothness',
           'worst compactness', 'worst concavity', 'worst concave points',
           'worst symmetry', 'worst fractal dimension'], dtype='<U23')




```python
df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_feat.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 569 entries, 0 to 568
    Data columns (total 30 columns):
    mean radius                569 non-null float64
    mean texture               569 non-null float64
    mean perimeter             569 non-null float64
    mean area                  569 non-null float64
    mean smoothness            569 non-null float64
    mean compactness           569 non-null float64
    mean concavity             569 non-null float64
    mean concave points        569 non-null float64
    mean symmetry              569 non-null float64
    mean fractal dimension     569 non-null float64
    radius error               569 non-null float64
    texture error              569 non-null float64
    perimeter error            569 non-null float64
    area error                 569 non-null float64
    smoothness error           569 non-null float64
    compactness error          569 non-null float64
    concavity error            569 non-null float64
    concave points error       569 non-null float64
    symmetry error             569 non-null float64
    fractal dimension error    569 non-null float64
    worst radius               569 non-null float64
    worst texture              569 non-null float64
    worst perimeter            569 non-null float64
    worst area                 569 non-null float64
    worst smoothness           569 non-null float64
    worst compactness          569 non-null float64
    worst concavity            569 non-null float64
    worst concave points       569 non-null float64
    worst symmetry             569 non-null float64
    worst fractal dimension    569 non-null float64
    dtypes: float64(30)
    memory usage: 133.4 KB



```python
df_feat.head()
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
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
df_target = pd.DataFrame(cancer['target'],columns=['Cancer'])
```


```python
df_target.head(50)
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
      <th>Cancer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>1</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# EDA
## Train Test Split


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.30, random_state=101)
```

## Train SVM


```python
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
```

    /Users/luoqiaoen/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)





    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)



## Predict and Evaluate


```python
predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
```

    [[  0  66]
     [  0 105]]



```python
print(classification_report(y_test,predictions))
```

                  precision    recall  f1-score   support
    
               0       0.00      0.00      0.00        66
               1       0.61      1.00      0.76       105
    
       micro avg       0.61      0.61      0.61       171
       macro avg       0.31      0.50      0.38       171
    weighted avg       0.38      0.61      0.47       171
    


    /Users/luoqiaoen/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/luoqiaoen/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/luoqiaoen/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


Notice that we are classifying everything into a single class! This means our model needs to have it parameters adjusted.

# GridSearch
We do some grid search to find hyper parameters


```python
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
```


```python
from sklearn.model_selection import GridSearchCV
```


```python
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
```


```python
grid.fit(X_train,y_train)
```

    /Users/luoqiaoen/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
      warnings.warn(CV_WARNING, FutureWarning)
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.0s remaining:    0.0s


    Fitting 3 folds for each of 25 candidates, totalling 75 fits
    [CV] C=0.1, gamma=1, kernel=rbf ......................................
    [CV]  C=0.1, gamma=1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=0.1, gamma=1, kernel=rbf ......................................
    [CV]  C=0.1, gamma=1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=0.1, gamma=1, kernel=rbf ......................................
    [CV]  C=0.1, gamma=1, kernel=rbf, score=0.6363636363636364, total=   0.0s
    [CV] C=0.1, gamma=0.1, kernel=rbf ....................................
    [CV]  C=0.1, gamma=0.1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=0.1, gamma=0.1, kernel=rbf ....................................
    [CV]  C=0.1, gamma=0.1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=0.1, gamma=0.1, kernel=rbf ....................................
    [CV]  C=0.1, gamma=0.1, kernel=rbf, score=0.6363636363636364, total=   0.0s
    [CV] C=0.1, gamma=0.01, kernel=rbf ...................................
    [CV]  C=0.1, gamma=0.01, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=0.1, gamma=0.01, kernel=rbf ...................................
    [CV]  C=0.1, gamma=0.01, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=0.1, gamma=0.01, kernel=rbf ...................................
    [CV]  C=0.1, gamma=0.01, kernel=rbf, score=0.6363636363636364, total=   0.0s
    [CV] C=0.1, gamma=0.001, kernel=rbf ..................................
    [CV]  C=0.1, gamma=0.001, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=0.1, gamma=0.001, kernel=rbf ..................................
    [CV]  C=0.1, gamma=0.001, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=0.1, gamma=0.001, kernel=rbf ..................................
    [CV]  C=0.1, gamma=0.001, kernel=rbf, score=0.6363636363636364, total=   0.0s
    [CV] C=0.1, gamma=0.0001, kernel=rbf .................................
    [CV]  C=0.1, gamma=0.0001, kernel=rbf, score=0.9022556390977443, total=   0.0s
    [CV] C=0.1, gamma=0.0001, kernel=rbf .................................
    [CV]  C=0.1, gamma=0.0001, kernel=rbf, score=0.9624060150375939, total=   0.0s
    [CV] C=0.1, gamma=0.0001, kernel=rbf .................................
    [CV]  C=0.1, gamma=0.0001, kernel=rbf, score=0.9166666666666666, total=   0.0s
    [CV] C=1, gamma=1, kernel=rbf ........................................
    [CV]  C=1, gamma=1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=1, gamma=1, kernel=rbf ........................................
    [CV]  C=1, gamma=1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=1, gamma=1, kernel=rbf ........................................
    [CV]  C=1, gamma=1, kernel=rbf, score=0.6363636363636364, total=   0.0s
    [CV] C=1, gamma=0.1, kernel=rbf ......................................
    [CV]  C=1, gamma=0.1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=1, gamma=0.1, kernel=rbf ......................................
    [CV]  C=1, gamma=0.1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=1, gamma=0.1, kernel=rbf ......................................
    [CV]  C=1, gamma=0.1, kernel=rbf, score=0.6363636363636364, total=   0.0s
    [CV] C=1, gamma=0.01, kernel=rbf .....................................
    [CV]  C=1, gamma=0.01, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=1, gamma=0.01, kernel=rbf .....................................
    [CV]  C=1, gamma=0.01, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=1, gamma=0.01, kernel=rbf .....................................
    [CV]  C=1, gamma=0.01, kernel=rbf, score=0.6363636363636364, total=   0.0s
    [CV] C=1, gamma=0.001, kernel=rbf ....................................
    [CV]  C=1, gamma=0.001, kernel=rbf, score=0.9022556390977443, total=   0.0s
    [CV] C=1, gamma=0.001, kernel=rbf ....................................
    [CV]  C=1, gamma=0.001, kernel=rbf, score=0.9398496240601504, total=   0.0s
    [CV] C=1, gamma=0.001, kernel=rbf ....................................
    [CV]  C=1, gamma=0.001, kernel=rbf, score=0.9545454545454546, total=   0.0s
    [CV] C=1, gamma=0.0001, kernel=rbf ...................................
    [CV]  C=1, gamma=0.0001, kernel=rbf, score=0.9398496240601504, total=   0.0s
    [CV] C=1, gamma=0.0001, kernel=rbf ...................................
    [CV]  C=1, gamma=0.0001, kernel=rbf, score=0.9699248120300752, total=   0.0s
    [CV] C=1, gamma=0.0001, kernel=rbf ...................................
    [CV]  C=1, gamma=0.0001, kernel=rbf, score=0.946969696969697, total=   0.0s
    [CV] C=10, gamma=1, kernel=rbf .......................................
    [CV]  C=10, gamma=1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=10, gamma=1, kernel=rbf .......................................
    [CV]  C=10, gamma=1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=10, gamma=1, kernel=rbf .......................................
    [CV]  C=10, gamma=1, kernel=rbf, score=0.6363636363636364, total=   0.0s
    [CV] C=10, gamma=0.1, kernel=rbf .....................................
    [CV]  C=10, gamma=0.1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=10, gamma=0.1, kernel=rbf .....................................
    [CV]  C=10, gamma=0.1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=10, gamma=0.1, kernel=rbf .....................................
    [CV]  C=10, gamma=0.1, kernel=rbf, score=0.6363636363636364, total=   0.0s
    [CV] C=10, gamma=0.01, kernel=rbf ....................................
    [CV]  C=10, gamma=0.01, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=10, gamma=0.01, kernel=rbf ....................................
    [CV]  C=10, gamma=0.01, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=10, gamma=0.01, kernel=rbf ....................................
    [CV]  C=10, gamma=0.01, kernel=rbf, score=0.6363636363636364, total=   0.0s
    [CV] C=10, gamma=0.001, kernel=rbf ...................................
    [CV]  C=10, gamma=0.001, kernel=rbf, score=0.8947368421052632, total=   0.0s
    [CV] C=10, gamma=0.001, kernel=rbf ...................................
    [CV]  C=10, gamma=0.001, kernel=rbf, score=0.9323308270676691, total=   0.0s
    [CV] C=10, gamma=0.001, kernel=rbf ...................................
    [CV]  C=10, gamma=0.001, kernel=rbf, score=0.9166666666666666, total=   0.0s
    [CV] C=10, gamma=0.0001, kernel=rbf ..................................
    [CV]  C=10, gamma=0.0001, kernel=rbf, score=0.9323308270676691, total=   0.0s
    [CV] C=10, gamma=0.0001, kernel=rbf ..................................
    [CV]  C=10, gamma=0.0001, kernel=rbf, score=0.9699248120300752, total=   0.0s
    [CV] C=10, gamma=0.0001, kernel=rbf ..................................
    [CV]  C=10, gamma=0.0001, kernel=rbf, score=0.9621212121212122, total=   0.0s
    [CV] C=100, gamma=1, kernel=rbf ......................................
    [CV]  C=100, gamma=1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=100, gamma=1, kernel=rbf ......................................
    [CV]  C=100, gamma=1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=100, gamma=1, kernel=rbf ......................................
    [CV]  C=100, gamma=1, kernel=rbf, score=0.6363636363636364, total=   0.0s
    [CV] C=100, gamma=0.1, kernel=rbf ....................................
    [CV]  C=100, gamma=0.1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=100, gamma=0.1, kernel=rbf ....................................
    [CV]  C=100, gamma=0.1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=100, gamma=0.1, kernel=rbf ....................................
    [CV]  C=100, gamma=0.1, kernel=rbf, score=0.6363636363636364, total=   0.0s
    [CV] C=100, gamma=0.01, kernel=rbf ...................................
    [CV]  C=100, gamma=0.01, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=100, gamma=0.01, kernel=rbf ...................................
    [CV]  C=100, gamma=0.01, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=100, gamma=0.01, kernel=rbf ...................................
    [CV]  C=100, gamma=0.01, kernel=rbf, score=0.6363636363636364, total=   0.0s
    [CV] C=100, gamma=0.001, kernel=rbf ..................................
    [CV]  C=100, gamma=0.001, kernel=rbf, score=0.8947368421052632, total=   0.0s
    [CV] C=100, gamma=0.001, kernel=rbf ..................................
    [CV]  C=100, gamma=0.001, kernel=rbf, score=0.9323308270676691, total=   0.0s
    [CV] C=100, gamma=0.001, kernel=rbf ..................................
    [CV]  C=100, gamma=0.001, kernel=rbf, score=0.9166666666666666, total=   0.0s
    [CV] C=100, gamma=0.0001, kernel=rbf .................................
    [CV]  C=100, gamma=0.0001, kernel=rbf, score=0.9172932330827067, total=   0.0s
    [CV] C=100, gamma=0.0001, kernel=rbf .................................
    [CV]  C=100, gamma=0.0001, kernel=rbf, score=0.9774436090225563, total=   0.0s
    [CV] C=100, gamma=0.0001, kernel=rbf .................................
    [CV]  C=100, gamma=0.0001, kernel=rbf, score=0.9393939393939394, total=   0.0s
    [CV] C=1000, gamma=1, kernel=rbf .....................................
    [CV]  C=1000, gamma=1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=1000, gamma=1, kernel=rbf .....................................
    [CV]  C=1000, gamma=1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=1000, gamma=1, kernel=rbf .....................................
    [CV]  C=1000, gamma=1, kernel=rbf, score=0.6363636363636364, total=   0.0s
    [CV] C=1000, gamma=0.1, kernel=rbf ...................................
    [CV]  C=1000, gamma=0.1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=1000, gamma=0.1, kernel=rbf ...................................
    [CV]  C=1000, gamma=0.1, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=1000, gamma=0.1, kernel=rbf ...................................
    [CV]  C=1000, gamma=0.1, kernel=rbf, score=0.6363636363636364, total=   0.0s
    [CV] C=1000, gamma=0.01, kernel=rbf ..................................
    [CV]  C=1000, gamma=0.01, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=1000, gamma=0.01, kernel=rbf ..................................
    [CV]  C=1000, gamma=0.01, kernel=rbf, score=0.631578947368421, total=   0.0s
    [CV] C=1000, gamma=0.01, kernel=rbf ..................................
    [CV]  C=1000, gamma=0.01, kernel=rbf, score=0.6363636363636364, total=   0.0s
    [CV] C=1000, gamma=0.001, kernel=rbf .................................
    [CV]  C=1000, gamma=0.001, kernel=rbf, score=0.8947368421052632, total=   0.0s
    [CV] C=1000, gamma=0.001, kernel=rbf .................................
    [CV]  C=1000, gamma=0.001, kernel=rbf, score=0.9323308270676691, total=   0.0s
    [CV] C=1000, gamma=0.001, kernel=rbf .................................
    [CV]  C=1000, gamma=0.001, kernel=rbf, score=0.9166666666666666, total=   0.0s
    [CV] C=1000, gamma=0.0001, kernel=rbf ................................
    [CV]  C=1000, gamma=0.0001, kernel=rbf, score=0.9097744360902256, total=   0.0s
    [CV] C=1000, gamma=0.0001, kernel=rbf ................................
    [CV]  C=1000, gamma=0.0001, kernel=rbf, score=0.9699248120300752, total=   0.0s
    [CV] C=1000, gamma=0.0001, kernel=rbf ................................
    [CV]  C=1000, gamma=0.0001, kernel=rbf, score=0.9318181818181818, total=   0.0s


    [Parallel(n_jobs=1)]: Done  75 out of  75 | elapsed:    0.9s finished





    GridSearchCV(cv='warn', error_score='raise-deprecating',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=3)




```python
grid.best_params_
```




    {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}




```python
grid.best_estimator_
```




    SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
grid_predictions = grid.predict(X_test)
```


```python
print(confusion_matrix(y_test,grid_predictions))
```

    [[ 60   6]
     [  3 102]]



```python
print(classification_report(y_test,grid_predictions))
```

                  precision    recall  f1-score   support
    
               0       0.95      0.91      0.93        66
               1       0.94      0.97      0.96       105
    
       micro avg       0.95      0.95      0.95       171
       macro avg       0.95      0.94      0.94       171
    weighted avg       0.95      0.95      0.95       171
    



```python

```
