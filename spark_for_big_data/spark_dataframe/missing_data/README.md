
# Missing Data

Often data sources are incomplete, which means you will have missing data, you have 3 basic options for filling in missing data (you will personally have to make the decision for what is the right approach:

* Just keep the missing data points.
* Drop them missing data points (including the entire row)
* Fill them in with some other value.

## Keeping the missing data
A few machine learning algorithms can easily deal with missing data, let's see what it looks like:


```python
from pyspark.sql import SparkSession
# May take a little while on a local computer
spark = SparkSession.builder.appName("missingdata").getOrCreate()
```


```python
df = spark.read.csv("ContainsNull.csv",header=True,inferSchema=True)
df.show()
```

    +----+-----+-----+
    |  Id| Name|Sales|
    +----+-----+-----+
    |emp1| John| null|
    |emp2| null| null|
    |emp3| null|345.0|
    |emp4|Cindy|456.0|
    +----+-----+-----+
    


## Drop the missing data

You can use the .na functions for missing data. The drop command has the following parameters:

    df.na.drop(how='any', thresh=None, subset=None)
    
    * param how: 'any' or 'all'.
    
        If 'any', drop a row if it contains any nulls.
        If 'all', drop a row only if all its values are null.
    
    * param thresh: int, default None
    
        If specified, drop rows that have less than `thresh` non-null values.
        This overwrites the `how` parameter.
        
    * param subset: 
        optional list of column names to consider.


```python
# Drop any row that contains missing data
df.na.drop().show()
```

    +----+-----+-----+
    |  Id| Name|Sales|
    +----+-----+-----+
    |emp4|Cindy|456.0|
    +----+-----+-----+
    



```python
# Has to have at least 2 NON-null values
df.na.drop(thresh=2).show()
```

    +----+-----+-----+
    |  Id| Name|Sales|
    +----+-----+-----+
    |emp1| John| null|
    |emp3| null|345.0|
    |emp4|Cindy|456.0|
    +----+-----+-----+
    



```python
df.na.drop(subset=["Sales"]).show()
```

    +----+-----+-----+
    |  Id| Name|Sales|
    +----+-----+-----+
    |emp3| null|345.0|
    |emp4|Cindy|456.0|
    +----+-----+-----+
    



```python
df.na.drop(how='all').show()
```

    +----+-----+-----+
    |  Id| Name|Sales|
    +----+-----+-----+
    |emp1| John| null|
    |emp2| null| null|
    |emp3| null|345.0|
    |emp4|Cindy|456.0|
    +----+-----+-----+
    


## Fill Values


```python
df.na.fill('NEW VALUE').show()
```

    +----+---------+-----+
    |  Id|     Name|Sales|
    +----+---------+-----+
    |emp1|     John| null|
    |emp2|NEW VALUE| null|
    |emp3|NEW VALUE|345.0|
    |emp4|    Cindy|456.0|
    +----+---------+-----+
    



```python
df.na.fill(0).show()
```

    +----+-----+-----+
    |  Id| Name|Sales|
    +----+-----+-----+
    |emp1| John|  0.0|
    |emp2| null|  0.0|
    |emp3| null|345.0|
    |emp4|Cindy|456.0|
    +----+-----+-----+
    



```python
df.na.fill('No Name',subset=['Name']).show()
```

    +----+-------+-----+
    |  Id|   Name|Sales|
    +----+-------+-----+
    |emp1|   John| null|
    |emp2|No Name| null|
    |emp3|No Name|345.0|
    |emp4|  Cindy|456.0|
    +----+-------+-----+
    


Fill values with mean


```python
from pyspark.sql.functions import mean
mean_val = df.select(mean(df['Sales'])).collect()

mean_val
```




    [Row(avg(Sales)=400.5)]




```python
mean_sales = mean_val[0][0] # Weird nested formatting of Row object!
```


```python
df.na.fill(mean_sales,["Sales"]).show()
```

    +----+-----+-----+
    |  Id| Name|Sales|
    +----+-----+-----+
    |emp1| John|400.5|
    |emp2| null|400.5|
    |emp3| null|345.0|
    |emp4|Cindy|456.0|
    +----+-----+-----+
    



```python
df.na.fill(df.select(mean(df['Sales'])).collect()[0][0],['Sales']).show() #oneliner
```

    +----+-----+-----+
    |  Id| Name|Sales|
    +----+-----+-----+
    |emp1| John|400.5|
    |emp2| null|400.5|
    |emp3| null|345.0|
    |emp4|Cindy|456.0|
    +----+-----+-----+
    



```python

```
