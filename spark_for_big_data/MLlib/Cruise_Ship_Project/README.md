
## Estimate how many crew members a ship will require
Data is:    
    
    Description: Measurements of ship size, capacity, crew, and age for 158 cruise
    ships.
    
    Variables/Columns
    Ship Name     1-20
    Cruise Line   21-40
    Age (as of 2013)   46-48
    Tonnage (1000s of tons)   50-56
    passengers (100s)   58-64
    Length (100s of feet)  66-72
    Cabins  (100s)   74-80
    Passenger Density   82-88
    Crew  (100s)   90-96


```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('cruise').getOrCreate()
df = spark.read.csv('cruise_ship_info.csv', inferSchema = True, header= True)
df.printSchema()
```

    root
     |-- Ship_name: string (nullable = true)
     |-- Cruise_line: string (nullable = true)
     |-- Age: integer (nullable = true)
     |-- Tonnage: double (nullable = true)
     |-- passengers: double (nullable = true)
     |-- length: double (nullable = true)
     |-- cabins: double (nullable = true)
     |-- passenger_density: double (nullable = true)
     |-- crew: double (nullable = true)
    



```python
df.show(1)
```

    +---------+-----------+---+------------------+----------+------+------+-----------------+----+
    |Ship_name|Cruise_line|Age|           Tonnage|passengers|length|cabins|passenger_density|crew|
    +---------+-----------+---+------------------+----------+------+------+-----------------+----+
    |  Journey|    Azamara|  6|30.276999999999997|      6.94|  5.94|  3.55|            42.64|3.55|
    +---------+-----------+---+------------------+----------+------+------+-----------------+----+
    only showing top 1 row
    


### change cruise line name into categorical variable


```python
df.groupBy('Cruise_line').count().show()
```

    +-----------------+-----+
    |      Cruise_line|count|
    +-----------------+-----+
    |            Costa|   11|
    |              P&O|    6|
    |           Cunard|    3|
    |Regent_Seven_Seas|    5|
    |              MSC|    8|
    |         Carnival|   22|
    |          Crystal|    2|
    |           Orient|    1|
    |         Princess|   17|
    |        Silversea|    4|
    |         Seabourn|    3|
    | Holland_American|   14|
    |         Windstar|    3|
    |           Disney|    2|
    |        Norwegian|   13|
    |          Oceania|    3|
    |          Azamara|    2|
    |        Celebrity|   10|
    |             Star|    6|
    |  Royal_Caribbean|   23|
    +-----------------+-----+
    



```python
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="Cruise_line", outputCol="cruise_cat")
indexed = indexer.fit(df).transform(df)
indexed.head(3)
```




    [Row(Ship_name='Journey', Cruise_line='Azamara', Age=6, Tonnage=30.276999999999997, passengers=6.94, length=5.94, cabins=3.55, passenger_density=42.64, crew=3.55, cruise_cat=16.0),
     Row(Ship_name='Quest', Cruise_line='Azamara', Age=6, Tonnage=30.276999999999997, passengers=6.94, length=5.94, cabins=3.55, passenger_density=42.64, crew=3.55, cruise_cat=16.0),
     Row(Ship_name='Celebration', Cruise_line='Carnival', Age=26, Tonnage=47.262, passengers=14.86, length=7.22, cabins=7.43, passenger_density=31.8, crew=6.7, cruise_cat=1.0)]




```python
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
indexed.columns
```




    ['Ship_name',
     'Cruise_line',
     'Age',
     'Tonnage',
     'passengers',
     'length',
     'cabins',
     'passenger_density',
     'crew',
     'cruise_cat']




```python
assembler = VectorAssembler(
  inputCols=['Age',
             'Tonnage',
             'passengers',
             'length',
             'cabins',
             'passenger_density',
             'cruise_cat'],
    outputCol="features")
```


```python
output = assembler.transform(indexed)
output.select("features", "crew").show()
```

    +--------------------+----+
    |            features|crew|
    +--------------------+----+
    |[6.0,30.276999999...|3.55|
    |[6.0,30.276999999...|3.55|
    |[26.0,47.262,14.8...| 6.7|
    |[11.0,110.0,29.74...|19.1|
    |[17.0,101.353,26....|10.0|
    |[22.0,70.367,20.5...| 9.2|
    |[15.0,70.367,20.5...| 9.2|
    |[23.0,70.367,20.5...| 9.2|
    |[19.0,70.367,20.5...| 9.2|
    |[6.0,110.23899999...|11.5|
    |[10.0,110.0,29.74...|11.6|
    |[28.0,46.052,14.5...| 6.6|
    |[18.0,70.367,20.5...| 9.2|
    |[17.0,70.367,20.5...| 9.2|
    |[11.0,86.0,21.24,...| 9.3|
    |[8.0,110.0,29.74,...|11.6|
    |[9.0,88.5,21.24,9...|10.3|
    |[15.0,70.367,20.5...| 9.2|
    |[12.0,88.5,21.24,...| 9.3|
    |[20.0,70.367,20.5...| 9.2|
    +--------------------+----+
    only showing top 20 rows
    



```python
final_data = output.select("features", "crew")
```

## Train Test Split


```python
train_data,test_data = final_data.randomSplit([0.7,0.3])
```

## Fit LR Model


```python
from pyspark.ml.regression import LinearRegression
# Create a Linear Regression Model object
lr = LinearRegression(labelCol='crew')
# Fit the model to the data and call this model lrModel
lrModel = lr.fit(train_data)
# Print the coefficients and intercept for linear regression
print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,lrModel.intercept))
```

    Coefficients: [-0.0143827060267077,0.013550326807490166,-0.1472883897035077,0.4421658751199197,0.8004711671410434,-0.012660035461297986,0.05622491964997087] Intercept: -0.7421882219140752



```python
test_results = lrModel.evaluate(test_data)
print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))
print("R2: {}".format(test_results.r2))
```

    RMSE: 1.4031637714149126
    MSE: 1.968868569411321
    R2: 0.8685127932752092



```python
from pyspark.sql.functions import corr
df.select(corr('crew','passengers')).show()
```

    +----------------------+
    |corr(crew, passengers)|
    +----------------------+
    |    0.9152341306065384|
    +----------------------+
    



```python
df.select(corr('crew','cabins')).show()
```

    +------------------+
    |corr(crew, cabins)|
    +------------------+
    |0.9508226063578497|
    +------------------+
    

