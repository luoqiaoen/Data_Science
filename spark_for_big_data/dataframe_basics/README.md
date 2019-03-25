
## Creating a DataFrame

First we need to start a SparkSession:


```python
from pyspark.sql import SparkSession
```


```python
# start a spark session
spark = SparkSession.builder.appName("Basics").getOrCreate()
```


```python
df = spark.read.json('people.json')
```


```python
df.show()
```

    +----+-------+
    | age|   name|
    +----+-------+
    |null|Michael|
    |  30|   Andy|
    |  19| Justin|
    +----+-------+
    


there is a null


```python
df.printSchema()
```

    root
     |-- age: long (nullable = true)
     |-- name: string (nullable = true)
    



```python
df.columns
```




    ['age', 'name']




```python
df.describe()
```




    DataFrame[summary: string, age: string, name: string]



### We will show we can build schema if the  .read method that doesn't have inferSchema() 


```python
from pyspark.sql.types import StructField,StringType,IntegerType,StructType
```

Next we need to create the list of Structure fields
    * :param name: string, name of the field.
    * :param dataType: :class:`DataType` of the field.
    * :param nullable: boolean, whether the field can be null (None) or not.


```python
data_schema = [StructField("age", IntegerType(), True),StructField("name", StringType(), True)]
```


```python
final_struc = StructType(fields=data_schema)
```


```python
df = spark.read.json('people.json', schema=final_struc)
```


```python
df.printSchema()
```

    root
     |-- age: integer (nullable = true)
     |-- name: string (nullable = true)
    


### Grabbing the data


```python
df['age']
```




    Column<b'age'>




```python
type(df['age'])
```




    pyspark.sql.column.Column




```python
df.select('age')
```




    DataFrame[age: int]




```python
type(df.select('age'))
```




    pyspark.sql.dataframe.DataFrame




```python
df.select('age').show()
```

    +----+
    | age|
    +----+
    |null|
    |  30|
    |  19|
    +----+
    



```python
# Returns list of Row objects
df.head(2)
```




    [Row(age=None, name='Michael'), Row(age=30, name='Andy')]



multiple columns


```python
df.select(['age','name'])
```




    DataFrame[age: int, name: string]




```python
df.select(['age','name']).show()
```

    +----+-------+
    | age|   name|
    +----+-------+
    |null|Michael|
    |  30|   Andy|
    |  19| Justin|
    +----+-------+
    


### Creating new columns


```python
df.withColumn('newage',df['age']).show() # copy age to newage
```

    +----+-------+------+
    | age|   name|newage|
    +----+-------+------+
    |null|Michael|  null|
    |  30|   Andy|    30|
    |  19| Justin|    19|
    +----+-------+------+
    



```python
df.show()
```

    +----+-------+
    | age|   name|
    +----+-------+
    |null|Michael|
    |  30|   Andy|
    |  19| Justin|
    +----+-------+
    



```python
df.withColumnRenamed('age','supernewage').show() # Simple Rename
```

    +-----------+-------+
    |supernewage|   name|
    +-----------+-------+
    |       null|Michael|
    |         30|   Andy|
    |         19| Justin|
    +-----------+-------+
    



```python
df.withColumn('doubleage',df['age']*2).show()
```

    +----+-------+---------+
    | age|   name|doubleage|
    +----+-------+---------+
    |null|Michael|     null|
    |  30|   Andy|       60|
    |  19| Justin|       38|
    +----+-------+---------+
    



```python
df.withColumn('add_one_age',df['age']+1).show()
```

    +----+-------+-----------+
    | age|   name|add_one_age|
    +----+-------+-----------+
    |null|Michael|       null|
    |  30|   Andy|         31|
    |  19| Justin|         20|
    +----+-------+-----------+
    



```python
df.withColumn('half_age',df['age']/2).show()
```

    +----+-------+--------+
    | age|   name|half_age|
    +----+-------+--------+
    |null|Michael|    null|
    |  30|   Andy|    15.0|
    |  19| Justin|     9.5|
    +----+-------+--------+
    


### Using SQL


```python
# Register the DataFrame as a SQL temporary view
df.createOrReplaceTempView("people")
```


```python
sql_results = spark.sql("SELECT * FROM people")
```


```python

```
