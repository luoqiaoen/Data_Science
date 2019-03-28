
# Logistic Regression Project

Use variables to predict if a customer will likely to chun.

The data is saved as customer_churn.csv. Here are the fields and their definitions:

    Name : Name of the latest contact at Company
    Age: Customer Age
    Total_Purchase: Total Ads Purchased
    Account_Manager: Binary 0=No manager, 1= Account manager assigned
    Years: Totaly Years as a customer
    Num_sites: Number of websites that use the service.
    Onboard_date: Date that the name of the latest contact was onboarded
    Location: Client HQ Address
    Company: Name of Client Company


```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('churn').getOrCreate()
data = spark.read.csv('customer_churn.csv',inferSchema=True,
                     header=True)
data.printSchema()
```

    root
     |-- Names: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- Total_Purchase: double (nullable = true)
     |-- Account_Manager: integer (nullable = true)
     |-- Years: double (nullable = true)
     |-- Num_Sites: double (nullable = true)
     |-- Onboard_date: timestamp (nullable = true)
     |-- Location: string (nullable = true)
     |-- Company: string (nullable = true)
     |-- Churn: integer (nullable = true)
    



```python
data.describe().show()
```

    +-------+-------------+-----------------+-----------------+------------------+-----------------+------------------+--------------------+--------------------+-------------------+
    |summary|        Names|              Age|   Total_Purchase|   Account_Manager|            Years|         Num_Sites|            Location|             Company|              Churn|
    +-------+-------------+-----------------+-----------------+------------------+-----------------+------------------+--------------------+--------------------+-------------------+
    |  count|          900|              900|              900|               900|              900|               900|                 900|                 900|                900|
    |   mean|         null|41.81666666666667|10062.82403333334|0.4811111111111111| 5.27315555555555| 8.587777777777777|                null|                null|0.16666666666666666|
    | stddev|         null|6.127560416916251|2408.644531858096|0.4999208935073339|1.274449013194616|1.7648355920350969|                null|                null| 0.3728852122772358|
    |    min|   Aaron King|             22.0|            100.0|                 0|              1.0|               3.0|00103 Jeffrey Cre...|     Abbott-Thompson|                  0|
    |    max|Zachary Walsh|             65.0|         18026.01|                 1|             9.15|              14.0|Unit 9800 Box 287...|Zuniga, Clark and...|                  1|
    +-------+-------------+-----------------+-----------------+------------------+-----------------+------------------+--------------------+--------------------+-------------------+
    



```python
data.columns
```




    ['Names',
     'Age',
     'Total_Purchase',
     'Account_Manager',
     'Years',
     'Num_Sites',
     'Onboard_date',
     'Location',
     'Company',
     'Churn']




```python
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['Age',
 'Total_Purchase',
 'Account_Manager',
 'Years',
 'Num_Sites'],outputCol='features')
output = assembler.transform(data)
final_data = output.select('features','churn')
```

## Test Train Split


```python
train_churn,test_churn = final_data.randomSplit([0.7,0.3])
```

## Fit Model, Logistic Regression


```python
from pyspark.ml.classification import LogisticRegression
lr_churn = LogisticRegression(labelCol='churn')
fitted_churn_model = lr_churn.fit(train_churn)
training_sum = fitted_churn_model.summary
training_sum.predictions.describe().show()
```

    +-------+------------------+-------------------+
    |summary|             churn|         prediction|
    +-------+------------------+-------------------+
    |  count|               618|                618|
    |   mean|0.1715210355987055|0.13106796116504854|
    | stddev|0.3772689762384247|0.33774803634441386|
    |    min|               0.0|                0.0|
    |    max|               1.0|                1.0|
    +-------+------------------+-------------------+
    


## Evaluate


```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator
pred_and_labels = fitted_churn_model.evaluate(test_churn)
pred_and_labels.predictions.show()
```

    +--------------------+-----+--------------------+--------------------+----------+
    |            features|churn|       rawPrediction|         probability|prediction|
    +--------------------+-----+--------------------+--------------------+----------+
    |[22.0,11254.38,1....|    0|[4.16228024656365...|[0.98466675997281...|       0.0|
    |[25.0,9672.03,0.0...|    0|[4.57855091366884...|[0.98983462869922...|       0.0|
    |[26.0,8787.39,1.0...|    1|[0.11189605069477...|[0.52794486126014...|       0.0|
    |[28.0,11245.38,0....|    0|[3.81191604502635...|[0.97837231424272...|       0.0|
    |[29.0,10203.18,1....|    0|[3.46984621457211...|[0.96981751730913...|       0.0|
    |[29.0,13240.01,1....|    0|[6.39274143203054...|[0.99832913478632...|       0.0|
    |[30.0,6744.87,0.0...|    0|[3.35690529463837...|[0.96633023221196...|       0.0|
    |[30.0,8874.83,0.0...|    0|[3.07939651936113...|[0.95603482594198...|       0.0|
    |[30.0,10744.14,1....|    1|[1.54960066269859...|[0.82485604763971...|       0.0|
    |[31.0,12264.68,1....|    0|[3.25480644708176...|[0.96284544199849...|       0.0|
    |[32.0,8011.38,0.0...|    0|[1.85938465821332...|[0.86522520883055...|       0.0|
    |[32.0,8575.71,0.0...|    0|[3.57084122661976...|[0.97263758611666...|       0.0|
    |[32.0,9885.12,1.0...|    1|[1.59203390689585...|[0.83090206712048...|       0.0|
    |[32.0,10716.75,0....|    0|[4.34589228366175...|[0.98720587197549...|       0.0|
    |[32.0,11715.72,0....|    0|[3.22825877140624...|[0.96188396522928...|       0.0|
    |[32.0,12142.99,0....|    0|[5.58380123051499...|[0.99625581922749...|       0.0|
    |[32.0,12479.72,0....|    0|[4.47191519222849...|[0.98870365241586...|       0.0|
    |[33.0,7720.61,1.0...|    0|[1.25359000733168...|[0.77792068992204...|       0.0|
    |[33.0,7750.54,1.0...|    0|[3.92921607769552...|[0.98071995029178...|       0.0|
    |[33.0,12638.51,1....|    0|[3.52598681508071...|[0.97141819725470...|       0.0|
    +--------------------+-----+--------------------+--------------------+----------+
    only showing top 20 rows
    


## AUC


```python
churn_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                           labelCol='churn')
auc = churn_eval.evaluate(pred_and_labels.predictions)
auc
```




    0.8135981665393429



## New Data


```python
final_lr_model = lr_churn.fit(final_data)
new_customers = spark.read.csv('new_customers.csv',inferSchema=True,
                              header=True)
new_customers.printSchema()
```

    root
     |-- Names: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- Total_Purchase: double (nullable = true)
     |-- Account_Manager: integer (nullable = true)
     |-- Years: double (nullable = true)
     |-- Num_Sites: double (nullable = true)
     |-- Onboard_date: timestamp (nullable = true)
     |-- Location: string (nullable = true)
     |-- Company: string (nullable = true)
    



```python
test_new_customers = assembler.transform(new_customers)
test_new_customers.printSchema()
```

    root
     |-- Names: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- Total_Purchase: double (nullable = true)
     |-- Account_Manager: integer (nullable = true)
     |-- Years: double (nullable = true)
     |-- Num_Sites: double (nullable = true)
     |-- Onboard_date: timestamp (nullable = true)
     |-- Location: string (nullable = true)
     |-- Company: string (nullable = true)
     |-- features: vector (nullable = true)
    



```python
final_results = final_lr_model.transform(test_new_customers)
final_results.select('Company','prediction').show()
```

    +----------------+----------+
    |         Company|prediction|
    +----------------+----------+
    |        King Ltd|       0.0|
    |   Cannon-Benson|       1.0|
    |Barron-Robertson|       1.0|
    |   Sexton-Golden|       1.0|
    |        Wood LLC|       0.0|
    |   Parks-Robbins|       1.0|
    +----------------+----------+
    


The 1s are likely to churn.


```python

```
