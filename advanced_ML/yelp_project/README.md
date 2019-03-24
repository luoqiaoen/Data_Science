

```python
import numpy as np
import pandas as pd
yelp = pd.read_csv('yelp.csv')
yelp.head()
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
      <th>business_id</th>
      <th>date</th>
      <th>review_id</th>
      <th>stars</th>
      <th>text</th>
      <th>type</th>
      <th>user_id</th>
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9yKzy9PApeiPPOUJEtnvkg</td>
      <td>2011-01-26</td>
      <td>fWKvX83p0-ka4JS3dc6E5A</td>
      <td>5</td>
      <td>My wife took me here on my birthday for breakf...</td>
      <td>review</td>
      <td>rLtl8ZkDX5vH5nAx9C3q5Q</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ZRJwVLyzEJq1VAihDhYiow</td>
      <td>2011-07-27</td>
      <td>IjZ33sJrzXqU-0X6U8NwyA</td>
      <td>5</td>
      <td>I have no idea why some people give bad review...</td>
      <td>review</td>
      <td>0a2KyEL0d3Yb1V6aivbIuQ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6oRAC4uyJCsJl1X0WZpVSA</td>
      <td>2012-06-14</td>
      <td>IESLBzqUCLdSzSqm0eCSxQ</td>
      <td>4</td>
      <td>love the gyro plate. Rice is so good and I als...</td>
      <td>review</td>
      <td>0hT2KtfLiobPvh6cDC8JQg</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>_1QQZuf4zZOyFCvXc0o6Vg</td>
      <td>2010-05-27</td>
      <td>G-WvGaISbqqaMHlNnByodA</td>
      <td>5</td>
      <td>Rosie, Dakota, and I LOVE Chaparral Dog Park!!...</td>
      <td>review</td>
      <td>uZetl9T0NcROGOyFfughhg</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6ozycU1RpktNG2-1BroVtw</td>
      <td>2012-01-05</td>
      <td>1uJFq2r5QfJG_6ExMRCaGw</td>
      <td>5</td>
      <td>General Manager Scott Petello is a good egg!!!...</td>
      <td>review</td>
      <td>vYmM4KTsC8ZfQBg-j5MWkw</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
yelp.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 10 columns):
    business_id    10000 non-null object
    date           10000 non-null object
    review_id      10000 non-null object
    stars          10000 non-null int64
    text           10000 non-null object
    type           10000 non-null object
    user_id        10000 non-null object
    cool           10000 non-null int64
    useful         10000 non-null int64
    funny          10000 non-null int64
    dtypes: int64(4), object(6)
    memory usage: 781.3+ KB



```python
yelp.describe()
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
      <th>stars</th>
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.777500</td>
      <td>0.876800</td>
      <td>1.409300</td>
      <td>0.701300</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.214636</td>
      <td>2.067861</td>
      <td>2.336647</td>
      <td>1.907942</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>77.000000</td>
      <td>76.000000</td>
      <td>57.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Create a new column called "text length" which is the number of words in the text column.**


```python
yelp['text length'] = yelp['text'].apply(len)
```

# EDA


```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
%matplotlib inline
```


```python
g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')
```




    <seaborn.axisgrid.FacetGrid at 0x1a1b7f0cc0>




![png](yelp_project_files/yelp_project_7_1.png)



```python
sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1bd89c88>




![png](yelp_project_files/yelp_project_8_1.png)



```python
sns.countplot(x='stars',data=yelp,palette='rainbow')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1be4e358>




![png](yelp_project_files/yelp_project_9_1.png)



```python
stars = yelp.groupby('stars').mean()
stars
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
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
      <th>text length</th>
    </tr>
    <tr>
      <th>stars</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.576769</td>
      <td>1.604806</td>
      <td>1.056075</td>
      <td>826.515354</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.719525</td>
      <td>1.563107</td>
      <td>0.875944</td>
      <td>842.256742</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.788501</td>
      <td>1.306639</td>
      <td>0.694730</td>
      <td>758.498289</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.954623</td>
      <td>1.395916</td>
      <td>0.670448</td>
      <td>712.923142</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.944261</td>
      <td>1.381780</td>
      <td>0.608631</td>
      <td>624.999101</td>
    </tr>
  </tbody>
</table>
</div>




```python
stars.corr()
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
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
      <th>text length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cool</th>
      <td>1.000000</td>
      <td>-0.743329</td>
      <td>-0.944939</td>
      <td>-0.857664</td>
    </tr>
    <tr>
      <th>useful</th>
      <td>-0.743329</td>
      <td>1.000000</td>
      <td>0.894506</td>
      <td>0.699881</td>
    </tr>
    <tr>
      <th>funny</th>
      <td>-0.944939</td>
      <td>0.894506</td>
      <td>1.000000</td>
      <td>0.843461</td>
    </tr>
    <tr>
      <th>text length</th>
      <td>-0.857664</td>
      <td>0.699881</td>
      <td>0.843461</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1c004390>




![png](yelp_project_files/yelp_project_12_1.png)


## NLP Classification Task

**Create a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.**


```python
yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]
```


```python
yelp_class.head()
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
      <th>business_id</th>
      <th>date</th>
      <th>review_id</th>
      <th>stars</th>
      <th>text</th>
      <th>type</th>
      <th>user_id</th>
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
      <th>text length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9yKzy9PApeiPPOUJEtnvkg</td>
      <td>2011-01-26</td>
      <td>fWKvX83p0-ka4JS3dc6E5A</td>
      <td>5</td>
      <td>My wife took me here on my birthday for breakf...</td>
      <td>review</td>
      <td>rLtl8ZkDX5vH5nAx9C3q5Q</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>889</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ZRJwVLyzEJq1VAihDhYiow</td>
      <td>2011-07-27</td>
      <td>IjZ33sJrzXqU-0X6U8NwyA</td>
      <td>5</td>
      <td>I have no idea why some people give bad review...</td>
      <td>review</td>
      <td>0a2KyEL0d3Yb1V6aivbIuQ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1345</td>
    </tr>
    <tr>
      <th>3</th>
      <td>_1QQZuf4zZOyFCvXc0o6Vg</td>
      <td>2010-05-27</td>
      <td>G-WvGaISbqqaMHlNnByodA</td>
      <td>5</td>
      <td>Rosie, Dakota, and I LOVE Chaparral Dog Park!!...</td>
      <td>review</td>
      <td>uZetl9T0NcROGOyFfughhg</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>419</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6ozycU1RpktNG2-1BroVtw</td>
      <td>2012-01-05</td>
      <td>1uJFq2r5QfJG_6ExMRCaGw</td>
      <td>5</td>
      <td>General Manager Scott Petello is a good egg!!!...</td>
      <td>review</td>
      <td>vYmM4KTsC8ZfQBg-j5MWkw</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>469</td>
    </tr>
    <tr>
      <th>6</th>
      <td>zp713qNhx8d9KCJJnrw1xA</td>
      <td>2010-02-12</td>
      <td>riFQ3vxNpP4rWLk_CSri2A</td>
      <td>5</td>
      <td>Drop what you're doing and drive here. After I...</td>
      <td>review</td>
      <td>wFweIWhv2fREZV_dYkz_1g</td>
      <td>7</td>
      <td>7</td>
      <td>4</td>
      <td>1565</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = yelp_class['text']
y = yelp_class['stars']
```


```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
```


```python
X = cv.fit_transform(X)
```

## Train Test Split


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
```

## Train the Model


```python
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)
```




    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)



## Evaluation


```python
predictions = nb.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))
```

    [[159  69]
     [ 22 976]]
    
    
                  precision    recall  f1-score   support
    
               1       0.88      0.70      0.78       228
               5       0.93      0.98      0.96       998
    
       micro avg       0.93      0.93      0.93      1226
       macro avg       0.91      0.84      0.87      1226
    weighted avg       0.92      0.93      0.92      1226
    


## Using Text Pre-processing


```python
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
```

# Using Pipeline

## Train Test Split


```python
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
```


```python
# May take some time
pipeline.fit(X_train,y_train)
```




    Pipeline(memory=None,
         steps=[('bow', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words=None,
            strip_...f=False, use_idf=True)), ('classifier', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))])



## Prediction


```python
predictions = pipeline.predict(X_test)
```


```python
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
```

    [[  0 228]
     [  0 998]]
                  precision    recall  f1-score   support
    
               1       0.00      0.00      0.00       228
               5       0.81      1.00      0.90       998
    
       micro avg       0.81      0.81      0.81      1226
       macro avg       0.41      0.50      0.45      1226
    weighted avg       0.66      0.81      0.73      1226
    


    /Users/luoqiaoen/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/luoqiaoen/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/luoqiaoen/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


***Looks like Tf-Idf actually made things worse!***


```python

```
