
# Data Visualization Using Python
## Packages in this note:
* pandas
* matplotlib
* seaborn
* plotly

We will list some functions and show examples.

## Pandas
There are several plot types built-in to pandas, most of them statistical plots by nature:
* df.plot.area
* df.plot.barh
* df.plot.density
* df.plot.hist
* df.plot.line
* df.plot.scatter
* df.plot.bar
* df.plot.box
* df.plot.hexbin
* df.plot.kde
* df.plot.pie


```python
import numpy as np
import pandas as pd
%matplotlib inline

df1 = pd.read_csv('df1',index_col=0)
df2 = pd.read_csv('df2')
df3 = pd.read_csv('df3')

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
df1['A'].hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10a2b0890>




![png](Data%20Visualization_files/Data%20Visualization_2_1.png)



```python
df3[['a','b']].plot.box()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x115df89d0>




![png](Data%20Visualization_files/Data%20Visualization_3_1.png)


## Seaborn
1. The distplot shows the distribution of a univariate set of observations.



```python
import seaborn as sns
%matplotlib inline
tips = sns.load_dataset('tips')
tips.head()
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
      <th>total_bill</th>
      <th>tip</th>
      <th>sex</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>3.31</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>3.61</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.distplot(tips['total_bill'])
# Safe to ignore warnings

sns.distplot(tips['total_bill'],kde=False,bins=30)

```

    /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval





    <matplotlib.axes._subplots.AxesSubplot at 0x117e55590>




![png](Data%20Visualization_files/Data%20Visualization_6_2.png)


2. jointplot() allows you to basically match up two distplots for bivariate data. With your choice of what **kind** parameter to compare with:
      * “scatter”
      * “reg”
      * “resid”
      * “kde”
      * “hex”


```python
sns.jointplot(x='total_bill',y='tip',data=tips,kind='hex')
```




    <seaborn.axisgrid.JointGrid at 0x117e469d0>




![png](Data%20Visualization_files/Data%20Visualization_8_1.png)


3. pairplot will plot pairwise relationships across an entire dataframe (for the numerical columns) and supports a color hue argument (for categorical columns).


```python
sns.pairplot(tips)
```




    <seaborn.axisgrid.PairGrid at 0x118164450>




![png](Data%20Visualization_files/Data%20Visualization_10_1.png)


you can also toggle the hue for another dimension.


```python
sns.pairplot(tips,hue='sex',palette='coolwarm')
```




    <seaborn.axisgrid.PairGrid at 0x1181887d0>




![png](Data%20Visualization_files/Data%20Visualization_12_1.png)


3. categorical data plots:
    * factorplot
    * boxplot
    * violinplot
    * stripplot
    * swarmplot
    * barplot
    * countplot


```python
sns.boxplot(data=tips,palette='rainbow',orient='h')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1190b0a90>




![png](Data%20Visualization_files/Data%20Visualization_14_1.png)



```python
sns.boxplot(x="day", y="total_bill", hue="smoker",data=tips, palette="coolwarm")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1191dd810>




![png](Data%20Visualization_files/Data%20Visualization_15_1.png)



```python
sns.violinplot(x="day", y="total_bill", data=tips,hue='sex',split=True,palette='Set1')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1192f5710>




![png](Data%20Visualization_files/Data%20Visualization_16_1.png)


4. Matrix form for correlation data


```python
import seaborn as sns
%matplotlib inline
flights = sns.load_dataset('flights')
tips = sns.load_dataset('tips')

tips.corr()
sns.heatmap(tips.corr(),cmap='coolwarm',annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1194f6e90>




![png](Data%20Visualization_files/Data%20Visualization_18_1.png)



```python
flights.pivot_table(values='passengers',index='month',columns='year')
pvflights = flights.pivot_table(values='passengers',index='month',columns='year')
sns.heatmap(pvflights,cmap='magma',linecolor='white',linewidths=1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x119660b90>




![png](Data%20Visualization_files/Data%20Visualization_19_1.png)


## Plotly and Cufflinks
Plotly is a library that allows you to create interactive plots that you can use in dashboards or websites (you can save them as html files or static images).


```python
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.go_offline()
df = pd.DataFrame(np.random.randn(100,4),columns='A B C D'.split())
# examples:
df.iplot(kind='scatter',x='A',y='B',mode='markers',size=10)
```
