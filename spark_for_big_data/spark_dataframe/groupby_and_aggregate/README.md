

```python
from pyspark.sql import SparkSession
```


```python
# May take a little while on a local computer
spark = SparkSession.builder.appName("aggs").getOrCreate()
```


```python
df = spark.read.csv('sales_info.csv',inferSchema=True,header=True)
```


```python
df.printSchema()
```

    root
     |-- Company: string (nullable = true)
     |-- Person: string (nullable = true)
     |-- Sales: double (nullable = true)
    



```python
df.groupBy("Company")
```




    <pyspark.sql.group.GroupedData at 0x110a03908>




```python
# Mean
df.groupBy("Company").mean().show()
```

    +-------+-----------------+
    |Company|       avg(Sales)|
    +-------+-----------------+
    |   APPL|            370.0|
    |   GOOG|            220.0|
    |     FB|            610.0|
    |   MSFT|322.3333333333333|
    +-------+-----------------+
    



```python
# Count
df.groupBy("Company").count().show()
```

    +-------+-----+
    |Company|count|
    +-------+-----+
    |   APPL|    4|
    |   GOOG|    3|
    |     FB|    2|
    |   MSFT|    3|
    +-------+-----+
    



```python
# Max
df.groupBy("Company").max().show()
```

    +-------+----------+
    |Company|max(Sales)|
    +-------+----------+
    |   APPL|     750.0|
    |   GOOG|     340.0|
    |     FB|     870.0|
    |   MSFT|     600.0|
    +-------+----------+
    



```python
# Sum
df.groupBy("Company").sum().show()
```

    +-------+----------+
    |Company|sum(Sales)|
    +-------+----------+
    |   APPL|    1480.0|
    |   GOOG|     660.0|
    |     FB|    1220.0|
    |   MSFT|     967.0|
    +-------+----------+
    


Not all methods need a groupby call, instead you can just call the generalized .agg() method, that will call the aggregate across all rows in the dataframe column specified. It can take in arguments as a single column, or create multiple aggregate calls all at once using dictionary notation.



```python
df.agg({'Sales':'max'}).show()
```

    +----------+
    |max(Sales)|
    +----------+
    |     870.0|
    +----------+
    


let's check


```python
group_data = df.groupBy("Company")
```


```python
group_data.agg({'Sales':'max'}).show()
```

    +-------+----------+
    |Company|max(Sales)|
    +-------+----------+
    |   APPL|     750.0|
    |   GOOG|     340.0|
    |     FB|     870.0|
    |   MSFT|     600.0|
    +-------+----------+
    


## Function


```python
from pyspark.sql.functions import countDistinct, avg,stddev
```


```python
df.select(avg('Sales')).show()
```

    +-----------------+
    |       avg(Sales)|
    +-----------------+
    |360.5833333333333|
    +-----------------+
    



```python
df.select(avg('Sales').alias('Average Sale')).show()
```

    +-----------------+
    |     Average Sale|
    +-----------------+
    |360.5833333333333|
    +-----------------+
    



```python
df.select(stddev('Sales')).show()
```

    +------------------+
    |stddev_samp(Sales)|
    +------------------+
    |250.08742410799007|
    +------------------+
    



```python
from pyspark.sql.functions import format_number
sales_std = df.select(stddev('Sales').alias('std'))
```


```python
sales_std.show()
```

    +------------------+
    |               std|
    +------------------+
    |250.08742410799007|
    +------------------+
    



```python
sales_std.select(format_number('std',2)).show()
```

    +---------------------+
    |format_number(std, 2)|
    +---------------------+
    |               250.09|
    +---------------------+
    


## Order By


```python
# Ascending
df.orderBy("Sales").show()
```

    +-------+-------+-----+
    |Company| Person|Sales|
    +-------+-------+-----+
    |   GOOG|Charlie|120.0|
    |   MSFT|    Amy|124.0|
    |   APPL|  Linda|130.0|
    |   GOOG|    Sam|200.0|
    |   MSFT|Vanessa|243.0|
    |   APPL|   John|250.0|
    |   GOOG|  Frank|340.0|
    |     FB|  Sarah|350.0|
    |   APPL|  Chris|350.0|
    |   MSFT|   Tina|600.0|
    |   APPL|   Mike|750.0|
    |     FB|   Carl|870.0|
    +-------+-------+-----+
    



```python
# Descending call off the column itself.
df.orderBy(df["Sales"].desc()).show()
```

    +-------+-------+-----+
    |Company| Person|Sales|
    +-------+-------+-----+
    |     FB|   Carl|870.0|
    |   APPL|   Mike|750.0|
    |   MSFT|   Tina|600.0|
    |     FB|  Sarah|350.0|
    |   APPL|  Chris|350.0|
    |   GOOG|  Frank|340.0|
    |   APPL|   John|250.0|
    |   MSFT|Vanessa|243.0|
    |   GOOG|    Sam|200.0|
    |   APPL|  Linda|130.0|
    |   MSFT|    Amy|124.0|
    |   GOOG|Charlie|120.0|
    +-------+-------+-----+
    



```python

```
