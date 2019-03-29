

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('nlp').getOrCreate()
data = spark.read.csv("smsspamcollection/SMSSpamCollection",inferSchema=True,sep='\t')
data = data.withColumnRenamed('_c0','class').withColumnRenamed('_c1','text')
data.show()
```

    +-----+--------------------+
    |class|                text|
    +-----+--------------------+
    |  ham|Go until jurong p...|
    |  ham|Ok lar... Joking ...|
    | spam|Free entry in 2 a...|
    |  ham|U dun say so earl...|
    |  ham|Nah I don't think...|
    | spam|FreeMsg Hey there...|
    |  ham|Even my brother i...|
    |  ham|As per your reque...|
    | spam|WINNER!! As a val...|
    | spam|Had your mobile 1...|
    |  ham|I'm gonna be home...|
    | spam|SIX chances to wi...|
    | spam|URGENT! You have ...|
    |  ham|I've been searchi...|
    |  ham|I HAVE A DATE ON ...|
    | spam|XXXMobileMovieClu...|
    |  ham|Oh k...i'm watchi...|
    |  ham|Eh u remember how...|
    |  ham|Fine if thats th...|
    | spam|England v Macedon...|
    +-----+--------------------+
    only showing top 20 rows
    


### Process Data


```python
from pyspark.sql.functions import length
data = data.withColumn('length',length(data['text']))
data.show()
```

    +-----+--------------------+------+
    |class|                text|length|
    +-----+--------------------+------+
    |  ham|Go until jurong p...|   111|
    |  ham|Ok lar... Joking ...|    29|
    | spam|Free entry in 2 a...|   155|
    |  ham|U dun say so earl...|    49|
    |  ham|Nah I don't think...|    61|
    | spam|FreeMsg Hey there...|   147|
    |  ham|Even my brother i...|    77|
    |  ham|As per your reque...|   160|
    | spam|WINNER!! As a val...|   157|
    | spam|Had your mobile 1...|   154|
    |  ham|I'm gonna be home...|   109|
    | spam|SIX chances to wi...|   136|
    | spam|URGENT! You have ...|   155|
    |  ham|I've been searchi...|   196|
    |  ham|I HAVE A DATE ON ...|    35|
    | spam|XXXMobileMovieClu...|   149|
    |  ham|Oh k...i'm watchi...|    26|
    |  ham|Eh u remember how...|    81|
    |  ham|Fine if thats th...|    56|
    | spam|England v Macedon...|   155|
    +-----+--------------------+------+
    only showing top 20 rows
    



```python
data.groupby('class').mean().show()
```

    +-----+-----------------+
    |class|      avg(length)|
    +-----+-----------------+
    |  ham|71.45431945307645|
    | spam|138.6706827309237|
    +-----+-----------------+
    


## Feature Transformations


```python
from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer

tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
idf = IDF(inputCol="c_vec", outputCol="tf_idf")
ham_spam_to_num = StringIndexer(inputCol='class',outputCol='label')
```


```python
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')
```

## Train The Model


```python
from pyspark.ml.classification import NaiveBayes
# Use defaults
nb = NaiveBayes()
```

## Pipeline


```python
from pyspark.ml import Pipeline
data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,count_vec,idf,clean_up])
cleaner = data_prep_pipe.fit(data)
clean_data = cleaner.transform(data)
```

## Evaluate


```python
clean_data = clean_data.select(['label','features'])
clean_data.show()
(training,testing) = clean_data.randomSplit([0.7,0.3])
spam_predictor = nb.fit(training)
data.printSchema()

test_results = spam_predictor.transform(testing)
test_results.show()
```

    +-----+--------------------+
    |label|            features|
    +-----+--------------------+
    |  0.0|(13424,[7,11,31,6...|
    |  0.0|(13424,[0,24,297,...|
    |  1.0|(13424,[2,13,19,3...|
    |  0.0|(13424,[0,70,80,1...|
    |  0.0|(13424,[36,134,31...|
    |  1.0|(13424,[10,60,139...|
    |  0.0|(13424,[10,53,103...|
    |  0.0|(13424,[125,184,4...|
    |  1.0|(13424,[1,47,118,...|
    |  1.0|(13424,[0,1,13,27...|
    |  0.0|(13424,[18,43,120...|
    |  1.0|(13424,[8,17,37,8...|
    |  1.0|(13424,[13,30,47,...|
    |  0.0|(13424,[39,96,217...|
    |  0.0|(13424,[552,1697,...|
    |  1.0|(13424,[30,109,11...|
    |  0.0|(13424,[82,214,47...|
    |  0.0|(13424,[0,2,49,13...|
    |  0.0|(13424,[0,74,105,...|
    |  1.0|(13424,[4,30,33,5...|
    +-----+--------------------+
    only showing top 20 rows
    
    root
     |-- class: string (nullable = true)
     |-- text: string (nullable = true)
     |-- length: integer (nullable = true)
    
    +-----+--------------------+--------------------+--------------------+----------+
    |label|            features|       rawPrediction|         probability|prediction|
    +-----+--------------------+--------------------+--------------------+----------+
    |  0.0|(13424,[0,1,2,13,...|[-608.04912984940...|[1.0,2.9379474528...|       0.0|
    |  0.0|(13424,[0,1,4,50,...|[-828.77559461619...|[1.0,1.9174950236...|       0.0|
    |  0.0|(13424,[0,1,7,8,1...|[-882.92688290822...|[1.0,1.2243506237...|       0.0|
    |  0.0|(13424,[0,1,14,78...|[-686.09193217471...|[1.0,1.7905643800...|       0.0|
    |  0.0|(13424,[0,1,17,19...|[-824.16339617698...|[1.0,9.5229871615...|       0.0|
    |  0.0|(13424,[0,1,21,27...|[-758.46122280222...|[1.0,9.1094670027...|       0.0|
    |  0.0|(13424,[0,1,24,31...|[-339.24541569978...|[1.0,1.6137446573...|       0.0|
    |  0.0|(13424,[0,1,27,35...|[-1472.1982530661...|[0.99999999999999...|       0.0|
    |  0.0|(13424,[0,1,27,88...|[-1524.1304685519...|[0.99999999999887...|       0.0|
    |  0.0|(13424,[0,1,72,10...|[-701.39267868147...|[1.0,1.1978935536...|       0.0|
    |  0.0|(13424,[0,1,146,1...|[-254.43030137635...|[0.22760950998709...|       1.0|
    |  0.0|(13424,[0,1,498,5...|[-317.74555857995...|[0.99999999999977...|       0.0|
    |  0.0|(13424,[0,1,3657,...|[-127.79001648235...|[0.99998163694761...|       0.0|
    |  0.0|(13424,[0,2,3,5,6...|[-2587.5496047962...|[1.0,4.0440002498...|       0.0|
    |  0.0|(13424,[0,2,3,6,9...|[-3306.6663161666...|[1.0,1.2049156682...|       0.0|
    |  0.0|(13424,[0,2,4,5,1...|[-1632.0633556873...|[1.0,1.6608937662...|       0.0|
    |  0.0|(13424,[0,2,4,8,2...|[-1407.9914261820...|[1.0,2.3695733543...|       0.0|
    |  0.0|(13424,[0,2,4,128...|[-638.93756207023...|[1.0,1.1606973591...|       0.0|
    |  0.0|(13424,[0,2,7,11,...|[-736.77804484568...|[1.0,7.2236300664...|       0.0|
    |  0.0|(13424,[0,2,7,114...|[-453.51834646355...|[1.0,2.3616093529...|       0.0|
    +-----+--------------------+--------------------+--------------------+----------+
    only showing top 20 rows
    



```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print("Accuracy of model at predicting spam was: {}".format(acc))
```

    Accuracy of model at predicting spam was: 0.9207647303347044



```python

```
