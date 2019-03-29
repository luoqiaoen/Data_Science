
## We will show

* A single decision tree
* A random forest
* A gradient boosted tree classifier
    
We will be using a college dataset to try to classify colleges as Private or Public based off these features:

    Private A factor with levels No and Yes indicating private or public university
    Apps Number of applications received
    Accept Number of applications accepted
    Enroll Number of new students enrolled
    Top10perc Pct. new students from top 10% of H.S. class
    Top25perc Pct. new students from top 25% of H.S. class
    F.Undergrad Number of fulltime undergraduates
    P.Undergrad Number of parttime undergraduates
    Outstate Out-of-state tuition
    Room.Board Room and board costs
    Books Estimated book costs
    Personal Estimated personal spending
    PhD Pct. of faculty with Ph.D.’s
    Terminal Pct. of faculty with terminal degree
    S.F.Ratio Student/faculty ratio
    perc.alumni Pct. alumni who donate
    Expend Instructional expenditure per student
    Grad.Rate Graduation rate


```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('treecode').getOrCreate()
data = spark.read.csv('College.csv',inferSchema=True,header=True)
data.printSchema()
```

    root
     |-- School: string (nullable = true)
     |-- Private: string (nullable = true)
     |-- Apps: integer (nullable = true)
     |-- Accept: integer (nullable = true)
     |-- Enroll: integer (nullable = true)
     |-- Top10perc: integer (nullable = true)
     |-- Top25perc: integer (nullable = true)
     |-- F_Undergrad: integer (nullable = true)
     |-- P_Undergrad: integer (nullable = true)
     |-- Outstate: integer (nullable = true)
     |-- Room_Board: integer (nullable = true)
     |-- Books: integer (nullable = true)
     |-- Personal: integer (nullable = true)
     |-- PhD: integer (nullable = true)
     |-- Terminal: integer (nullable = true)
     |-- S_F_Ratio: double (nullable = true)
     |-- perc_alumni: integer (nullable = true)
     |-- Expend: integer (nullable = true)
     |-- Grad_Rate: integer (nullable = true)
    


## Format the Data


```python
# It needs to be in the form of two columns
# ("label","features")

# Import VectorAssembler and Vectors
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
data.columns
```




    ['School',
     'Private',
     'Apps',
     'Accept',
     'Enroll',
     'Top10perc',
     'Top25perc',
     'F_Undergrad',
     'P_Undergrad',
     'Outstate',
     'Room_Board',
     'Books',
     'Personal',
     'PhD',
     'Terminal',
     'S_F_Ratio',
     'perc_alumni',
     'Expend',
     'Grad_Rate']




```python
assembler = VectorAssembler(
  inputCols=['Apps',
             'Accept',
             'Enroll',
             'Top10perc',
             'Top25perc',
             'F_Undergrad',
             'P_Undergrad',
             'Outstate',
             'Room_Board',
             'Books',
             'Personal',
             'PhD',
             'Terminal',
             'S_F_Ratio',
             'perc_alumni',
             'Expend',
             'Grad_Rate'],
              outputCol="features")
```


```python
output = assembler.transform(data)
```

Private column being "yes" or "no" needs to be transformed by StringIndexer


```python
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="Private", outputCol="PrivateIndex")
output_fixed = indexer.fit(output).transform(output)
final_data = output_fixed.select("features",'PrivateIndex')
train_data,test_data = final_data.randomSplit([0.7,0.3])
```

## The Classifier


```python
from pyspark.ml.classification import DecisionTreeClassifier,GBTClassifier,RandomForestClassifier
from pyspark.ml import Pipeline

# Three * A single decision tree * A random forest * A gradient boosted tree classifier

dtc = DecisionTreeClassifier(labelCol='PrivateIndex',featuresCol='features')
rfc = RandomForestClassifier(labelCol='PrivateIndex',featuresCol='features')
gbt = GBTClassifier(labelCol='PrivateIndex',featuresCol='features')
```


```python
# Train the models (its three models, so it might take some time)
dtc_model = dtc.fit(train_data)
rfc_model = rfc.fit(train_data)
gbt_model = gbt.fit(train_data)
```

## Model Compare


```python
dtc_predictions = dtc_model.transform(test_data)
rfc_predictions = rfc_model.transform(test_data)
gbt_predictions = gbt_model.transform(test_data)
```


```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

acc_evaluator = MulticlassClassificationEvaluator(labelCol="PrivateIndex", predictionCol="prediction", metricName="accuracy")

dtc_acc = acc_evaluator.evaluate(dtc_predictions)
rfc_acc = acc_evaluator.evaluate(rfc_predictions)
gbt_acc = acc_evaluator.evaluate(gbt_predictions)

print("Here are the results!")
print('-'*80)
print('A single decision tree had an accuracy of: {0:2.2f}%'.format(dtc_acc*100))
print('-'*80)
print('A random forest ensemble had an accuracy of: {0:2.2f}%'.format(rfc_acc*100))
print('-'*80)
print('A ensemble using GBT had an accuracy of: {0:2.2f}%'.format(gbt_acc*100))
```

    Here are the results!
    --------------------------------------------------------------------------------
    A single decision tree had an accuracy of: 90.58%
    --------------------------------------------------------------------------------
    A random forest ensemble had an accuracy of: 94.17%
    --------------------------------------------------------------------------------
    A ensemble using GBT had an accuracy of: 90.58%



```python

```
