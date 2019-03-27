
# K Means Clustering

Look at seeds: https://archive.ics.uci.edu/ml/datasets/seeds.

Attribute Information:

To construct the data, seven geometric parameters of wheat kernels were measured: 
1. area A, 
2. perimeter P, 
3. compactness C = 4*pi*A/P^2, 
4. length of kernel, 
5. width of kernel, 
6. asymmetry coefficient 
7. length of kernel groove. 


```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('cluster').getOrCreate()
```


```python
from pyspark.ml.clustering import KMeans

# Loads data.
dataset = spark.read.csv("seeds_dataset.csv",header=True,inferSchema=True)
```


```python
dataset.head()
```




    Row(area=15.26, perimeter=14.84, compactness=0.871, length_of_kernel=5.763, width_of_kernel=3.312, asymmetry_coefficient=2.221, length_of_groove=5.22)




```python
dataset.describe().show()
```

    +-------+------------------+------------------+--------------------+-------------------+------------------+---------------------+-------------------+
    |summary|              area|         perimeter|         compactness|   length_of_kernel|   width_of_kernel|asymmetry_coefficient|   length_of_groove|
    +-------+------------------+------------------+--------------------+-------------------+------------------+---------------------+-------------------+
    |  count|               210|               210|                 210|                210|               210|                  210|                210|
    |   mean|14.847523809523816|14.559285714285718|  0.8709985714285714|  5.628533333333335| 3.258604761904762|   3.7001999999999997|  5.408071428571429|
    | stddev|2.9096994306873647|1.3059587265640225|0.023629416583846364|0.44306347772644983|0.3777144449065867|   1.5035589702547392|0.49148049910240543|
    |    min|             10.59|             12.41|              0.8081|              4.899|              2.63|                0.765|              4.519|
    |    max|             21.18|             17.25|              0.9183|              6.675|             4.033|                8.456|               6.55|
    +-------+------------------+------------------+--------------------+-------------------+------------------+---------------------+-------------------+
    


## Format the Data


```python
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
```


```python
dataset.columns
```




    ['area',
     'perimeter',
     'compactness',
     'length_of_kernel',
     'width_of_kernel',
     'asymmetry_coefficient',
     'length_of_groove']




```python
vec_assembler = VectorAssembler(inputCols = dataset.columns, outputCol='features')
```


```python
final_data = vec_assembler.transform(dataset)
```


```python
final_data.show()
```

    +-----+---------+-----------+------------------+------------------+---------------------+------------------+--------------------+
    | area|perimeter|compactness|  length_of_kernel|   width_of_kernel|asymmetry_coefficient|  length_of_groove|            features|
    +-----+---------+-----------+------------------+------------------+---------------------+------------------+--------------------+
    |15.26|    14.84|      0.871|             5.763|             3.312|                2.221|              5.22|[15.26,14.84,0.87...|
    |14.88|    14.57|     0.8811| 5.553999999999999|             3.333|                1.018|             4.956|[14.88,14.57,0.88...|
    |14.29|    14.09|      0.905|             5.291|3.3369999999999997|                2.699|             4.825|[14.29,14.09,0.90...|
    |13.84|    13.94|     0.8955|             5.324|3.3789999999999996|                2.259|             4.805|[13.84,13.94,0.89...|
    |16.14|    14.99|     0.9034|5.6579999999999995|             3.562|                1.355|             5.175|[16.14,14.99,0.90...|
    |14.38|    14.21|     0.8951|             5.386|             3.312|   2.4619999999999997|             4.956|[14.38,14.21,0.89...|
    |14.69|    14.49|     0.8799|             5.563|             3.259|   3.5860000000000003| 5.218999999999999|[14.69,14.49,0.87...|
    |14.11|     14.1|     0.8911|              5.42|             3.302|                  2.7|               5.0|[14.11,14.1,0.891...|
    |16.63|    15.46|     0.8747|             6.053|             3.465|                 2.04| 5.877000000000001|[16.63,15.46,0.87...|
    |16.44|    15.25|      0.888|5.8839999999999995|             3.505|                1.969|5.5329999999999995|[16.44,15.25,0.88...|
    |15.26|    14.85|     0.8696|5.7139999999999995|             3.242|                4.543|             5.314|[15.26,14.85,0.86...|
    |14.03|    14.16|     0.8796|             5.438|             3.201|   1.7169999999999999|             5.001|[14.03,14.16,0.87...|
    |13.89|    14.02|      0.888|             5.439|             3.199|                3.986|             4.738|[13.89,14.02,0.88...|
    |13.78|    14.06|     0.8759|             5.479|             3.156|                3.136|             4.872|[13.78,14.06,0.87...|
    |13.74|    14.05|     0.8744|             5.482|             3.114|                2.932|             4.825|[13.74,14.05,0.87...|
    |14.59|    14.28|     0.8993|             5.351|             3.333|                4.185| 4.781000000000001|[14.59,14.28,0.89...|
    |13.99|    13.83|     0.9183|             5.119|             3.383|                5.234| 4.781000000000001|[13.99,13.83,0.91...|
    |15.69|    14.75|     0.9058|             5.527|             3.514|                1.599|             5.046|[15.69,14.75,0.90...|
    | 14.7|    14.21|     0.9153|             5.205|             3.466|                1.767|             4.649|[14.7,14.21,0.915...|
    |12.72|    13.57|     0.8686|             5.226|             3.049|                4.102|             4.914|[12.72,13.57,0.86...|
    +-----+---------+-----------+------------------+------------------+---------------------+------------------+--------------------+
    only showing top 20 rows
    


## Scale the data

[the curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)


```python
from pyspark.ml.feature import StandardScaler
```


```python
scaler = StandardScaler(inputCol = "features", outputCol = "scaledFeatures", withStd=True, withMean=False)
```


```python
scalerModel = scaler.fit(final_data)
```


```python
final_data = scalerModel.transform(final_data)
```

## Train Model


```python
kmeans = KMeans(featuresCol='scaledFeatures',k=3)
model = kmeans.fit(final_data)
```

## Evaluate
Within Set Sum of Squared Errors.


```python
wssse = model.computeCost(final_data)
print("Within Set Sum of Squared Errors = " + str(wssse))
```

    Within Set Sum of Squared Errors = 428.60820118716356



```python
# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
```

    Cluster Centers: 
    [ 4.96198582 10.97871333 37.30930808 12.44647267  8.62880781  1.80061978
     10.41913733]
    [ 4.07497225 10.14410142 35.89816849 11.80812742  7.54416916  3.15410901
     10.38031464]
    [ 6.35645488 12.40730852 37.41990178 13.93860446  9.7892399   2.41585013
     12.29286107]



```python
model.transform(final_data).select('prediction').show()
```

    +----------+
    |prediction|
    +----------+
    |         0|
    |         0|
    |         0|
    |         0|
    |         0|
    |         0|
    |         0|
    |         0|
    |         2|
    |         0|
    |         0|
    |         0|
    |         0|
    |         0|
    |         0|
    |         0|
    |         0|
    |         0|
    |         0|
    |         1|
    +----------+
    only showing top 20 rows
    


## Optimal K


```python
for i in range(19):
    kmeans = KMeans(featuresCol='scaledFeatures',k=i+2)
    model = kmeans.fit(final_data)
    wssse = model.computeCost(final_data)
    print(str(i+2)+":Within Set Sum of Squared Errors = " + str(wssse))
```

    2:Within Set Sum of Squared Errors = 656.7932253385325
    3:Within Set Sum of Squared Errors = 428.60820118716356
    4:Within Set Sum of Squared Errors = 380.89132510833224
    5:Within Set Sum of Squared Errors = 330.76275833275713
    6:Within Set Sum of Squared Errors = 298.3094949234943
    7:Within Set Sum of Squared Errors = 261.9450937266424
    8:Within Set Sum of Squared Errors = 257.0716512204947
    9:Within Set Sum of Squared Errors = 244.31641522926247
    10:Within Set Sum of Squared Errors = 213.24847477392632
    11:Within Set Sum of Squared Errors = 189.3210964863838
    12:Within Set Sum of Squared Errors = 194.6960479891054
    13:Within Set Sum of Squared Errors = 173.27820078129628
    14:Within Set Sum of Squared Errors = 165.55880281710859
    15:Within Set Sum of Squared Errors = 174.85103100863853
    16:Within Set Sum of Squared Errors = 153.99977599563772
    17:Within Set Sum of Squared Errors = 148.89703933579267
    18:Within Set Sum of Squared Errors = 138.39814097760518
    19:Within Set Sum of Squared Errors = 134.11526155592128
    20:Within Set Sum of Squared Errors = 133.33154240278918


We know there only 3 clusters, so this is definitely overfitting.


```python

```
