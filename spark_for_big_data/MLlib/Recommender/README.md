

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('rec').getOrCreate()
```

Spark MLlib library for Machine Learning provides a Collaborative Filtering implementation by using Alternating Least Squares. The implementation in MLlib has these parameters:

* numBlocks is the number of blocks used to parallelize computation (set to -1 to auto-configure).
* rank is the number of latent factors in the model.
* iterations is the number of iterations to run.
* lambda specifies the regularization parameter in ALS.
* implicitPrefs specifies whether to use the explicit feedback ALS variant or one adapted for implicit feedback data.
* alpha is a parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations.


```python
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
data = spark.read.csv('movielens_ratings.csv',inferSchema=True,header=True)
```


```python
data.head()
```




    Row(movieId=2, rating=3.0, userId=0)




```python
data.describe().show()
```

    +-------+------------------+------------------+------------------+
    |summary|           movieId|            rating|            userId|
    +-------+------------------+------------------+------------------+
    |  count|              1501|              1501|              1501|
    |   mean| 49.40572951365756|1.7741505662891406|14.383744170552964|
    | stddev|28.937034065088994| 1.187276166124803| 8.591040424293272|
    |    min|                 0|               1.0|                 0|
    |    max|                99|               5.0|                29|
    +-------+------------------+------------------+------------------+
    


## Train Test Split


```python
(training, test) = data.randomSplit([0.8, 0.2])
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
model = als.fit(training)
```


```python
predictions = model.transform(test)
predictions.show()
```

    +-------+------+------+----------+
    |movieId|rating|userId|prediction|
    +-------+------+------+----------+
    |     31|   1.0|    26| -1.466789|
    |     31|   3.0|     8| 2.4332948|
    |     85|   1.0|    12| 1.7107557|
    |     85|   1.0|    23|-1.8163849|
    |     85|   4.0|     7| 3.6244862|
    |     53|   5.0|     8|   0.90004|
    |     53|   1.0|     7| 2.0412302|
    |     53|   1.0|    25|-2.5514557|
    |     78|   1.0|     8|  1.485697|
    |     78|   1.0|    24| 1.2067804|
    |     34|   1.0|    14| 0.5680803|
    |     81|   1.0|     1|-1.0269912|
    |     81|   1.0|    21|  4.902738|
    |     28|   3.0|     1| 1.2324274|
    |     28|   5.0|    18| 0.4781452|
    |     76|   1.0|     1| 3.2650518|
    |     26|   1.0|     6|  1.803495|
    |     26|   1.0|    18|0.43796617|
    |     27|   1.0|     5| 2.7264397|
    |     27|   1.0|    15| 2.6615295|
    +-------+------+------+----------+
    only showing top 20 rows
    



```python
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
```

    Root-mean-square error = 1.926340354506699



```python
single_user = test.filter(test['userId']==11).select(['movieId','userId'])
single_user.show()
```

    +-------+------+
    |movieId|userId|
    +-------+------+
    |      6|    11|
    |      9|    11|
    |     13|    11|
    |     19|    11|
    |     22|    11|
    |     32|    11|
    |     37|    11|
    |     38|    11|
    |     41|    11|
    |     43|    11|
    |     75|    11|
    |     90|    11|
    +-------+------+
    



```python
reccomendations = model.transform(single_user)
reccomendations.orderBy('prediction',ascending=False).show()
```

    +-------+------+-----------+
    |movieId|userId| prediction|
    +-------+------+-----------+
    |     38|    11|  5.7198462|
    |     22|    11|   3.756042|
    |     43|    11|  3.3207028|
    |     19|    11|  2.9537303|
    |     75|    11|  2.3123062|
    |      6|    11|  2.0499704|
    |     32|    11|  1.8109373|
    |     90|    11|  1.6737628|
    |     13|    11|   1.343168|
    |     37|    11|  0.7733513|
    |      9|    11|-0.17359355|
    |     41|    11|-0.56808823|
    +-------+------+-----------+
    



```python

```
