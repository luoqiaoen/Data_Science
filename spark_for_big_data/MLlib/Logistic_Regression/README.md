

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('titanic').getOrCreate()
data = spark.read.csv('titanic.csv',inferSchema = True, header=True)
data.printSchema()
```

    root
     |-- PassengerId: integer (nullable = true)
     |-- Survived: integer (nullable = true)
     |-- Pclass: integer (nullable = true)
     |-- Name: string (nullable = true)
     |-- Sex: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- SibSp: integer (nullable = true)
     |-- Parch: integer (nullable = true)
     |-- Ticket: string (nullable = true)
     |-- Fare: double (nullable = true)
     |-- Cabin: string (nullable = true)
     |-- Embarked: string (nullable = true)
    



```python
data.columns
```




    ['PassengerId',
     'Survived',
     'Pclass',
     'Name',
     'Sex',
     'Age',
     'SibSp',
     'Parch',
     'Ticket',
     'Fare',
     'Cabin',
     'Embarked']




```python
my_cols = data.select(['Survived',
 'Pclass',
 'Sex',
 'Age',
 'SibSp',
 'Parch',
 'Fare',
 'Embarked'])
```


```python
my_cols.head(5)
```




    [Row(Survived=0, Pclass=3, Sex='male', Age=22.0, SibSp=1, Parch=0, Fare=7.25, Embarked='S'),
     Row(Survived=1, Pclass=1, Sex='female', Age=38.0, SibSp=1, Parch=0, Fare=71.2833, Embarked='C'),
     Row(Survived=1, Pclass=3, Sex='female', Age=26.0, SibSp=0, Parch=0, Fare=7.925, Embarked='S'),
     Row(Survived=1, Pclass=1, Sex='female', Age=35.0, SibSp=1, Parch=0, Fare=53.1, Embarked='S'),
     Row(Survived=0, Pclass=3, Sex='male', Age=35.0, SibSp=0, Parch=0, Fare=8.05, Embarked='S')]




```python
my_final_data = my_cols.na.drop()
```


```python
my_final_data.head(5)
```




    [Row(Survived=0, Pclass=3, Sex='male', Age=22.0, SibSp=1, Parch=0, Fare=7.25, Embarked='S'),
     Row(Survived=1, Pclass=1, Sex='female', Age=38.0, SibSp=1, Parch=0, Fare=71.2833, Embarked='C'),
     Row(Survived=1, Pclass=3, Sex='female', Age=26.0, SibSp=0, Parch=0, Fare=7.925, Embarked='S'),
     Row(Survived=1, Pclass=1, Sex='female', Age=35.0, SibSp=1, Parch=0, Fare=53.1, Embarked='S'),
     Row(Survived=0, Pclass=3, Sex='male', Age=35.0, SibSp=0, Parch=0, Fare=8.05, Embarked='S')]



## Categorical Columns


```python
from pyspark.ml.feature import (VectorAssembler, VectorIndexer, 
                                OneHotEncoder, StringIndexer)
gender_indexer = StringIndexer(inputCol='Sex',outputCol='SexIndex')
gender_encoder = OneHotEncoder(inputCol='SexIndex',outputCol='SexVec')
embark_indexer = StringIndexer(inputCol='Embarked',outputCol='EmbarkIndex')
embark_encoder = OneHotEncoder(inputCol='EmbarkIndex',outputCol='EmbarkVec')
assembler = VectorAssembler(inputCols=['Pclass',
 'SexVec',
 'Age',
 'SibSp',
 'Parch',
 'Fare',
 'EmbarkVec'],outputCol='features')
```

## Logistic Regression Pipeline


```python
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

log_reg_titanic = LogisticRegression(featuresCol='features',labelCol='Survived')
pipeline = Pipeline(stages=[gender_indexer,embark_indexer,
                           gender_encoder,embark_encoder,
                           assembler,log_reg_titanic])
train_titanic_data, test_titanic_data = my_final_data.randomSplit([0.7,.3])
fit_model = pipeline.fit(train_titanic_data)
results = fit_model.transform(test_titanic_data)
```


```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator
my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                       labelCol='Survived')
results.select('Survived','prediction').show()
```

    +--------+----------+
    |Survived|prediction|
    +--------+----------+
    |       0|       0.0|
    |       0|       1.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       1.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    +--------+----------+
    only showing top 20 rows
    



```python
AUC = my_eval.evaluate(results)
```


```python
AUC
```




    0.7962056303549572




```python

```
