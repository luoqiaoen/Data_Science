
## Start Spark Session


```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("walmart").getOrCreate()
```

## Load Data


```python
df = spark.read.csv('walmart_stock.csv', header = True, inferSchema = True)
df.columns
```




    ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']




```python
df.printSchema()
```

    root
     |-- Date: timestamp (nullable = true)
     |-- Open: double (nullable = true)
     |-- High: double (nullable = true)
     |-- Low: double (nullable = true)
     |-- Close: double (nullable = true)
     |-- Volume: integer (nullable = true)
     |-- Adj Close: double (nullable = true)
    


## Print out the first 5 rows


```python
for row in df.head(5):
    print(row)
    print('\n')
```

    Row(Date=datetime.datetime(2012, 1, 3, 0, 0), Open=59.970001, High=61.060001, Low=59.869999, Close=60.330002, Volume=12668800, Adj Close=52.619234999999996)
    
    
    Row(Date=datetime.datetime(2012, 1, 4, 0, 0), Open=60.209998999999996, High=60.349998, Low=59.470001, Close=59.709998999999996, Volume=9593300, Adj Close=52.078475)
    
    
    Row(Date=datetime.datetime(2012, 1, 5, 0, 0), Open=59.349998, High=59.619999, Low=58.369999, Close=59.419998, Volume=12768200, Adj Close=51.825539)
    
    
    Row(Date=datetime.datetime(2012, 1, 6, 0, 0), Open=59.419998, High=59.450001, Low=58.869999, Close=59.0, Volume=8069400, Adj Close=51.45922)
    
    
    Row(Date=datetime.datetime(2012, 1, 9, 0, 0), Open=59.029999, High=59.549999, Low=58.919998, Close=59.18, Volume=6679300, Adj Close=51.616215000000004)
    
    



```python
df.describe().show()
```

    +-------+------------------+-----------------+-----------------+-----------------+-----------------+-----------------+
    |summary|              Open|             High|              Low|            Close|           Volume|        Adj Close|
    +-------+------------------+-----------------+-----------------+-----------------+-----------------+-----------------+
    |  count|              1258|             1258|             1258|             1258|             1258|             1258|
    |   mean| 72.35785375357709|72.83938807631165| 71.9186009594594|72.38844998012726|8222093.481717011|67.23883848728146|
    | stddev|  6.76809024470826|6.768186808159218|6.744075756255496|6.756859163732991|  4519780.8431556|6.722609449996857|
    |    min|56.389998999999996|        57.060001|        56.299999|        56.419998|          2094900|        50.363689|
    |    max|         90.800003|        90.970001|            89.25|        90.470001|         80898100|84.91421600000001|
    +-------+------------------+-----------------+-----------------+-----------------+-----------------+-----------------+
    



```python
df.describe().printSchema()
```

    root
     |-- summary: string (nullable = true)
     |-- Open: string (nullable = true)
     |-- High: string (nullable = true)
     |-- Low: string (nullable = true)
     |-- Close: string (nullable = true)
     |-- Volume: string (nullable = true)
     |-- Adj Close: string (nullable = true)
    


### Convert String To Float


```python
from pyspark.sql.functions import format_number
result = df.describe()
result.select(result['summary'],
              format_number(result['Open'].cast('float'),2).alias('Open'),
              format_number(result['High'].cast('float'),2).alias('High'),
              format_number(result['Low'].cast('float'),2).alias('Low'),
              format_number(result['Close'].cast('float'),2).alias('Close'),
              result['Volume'].cast('int').alias('Volume')
             ).show()
```

    +-------+--------+--------+--------+--------+--------+
    |summary|    Open|    High|     Low|   Close|  Volume|
    +-------+--------+--------+--------+--------+--------+
    |  count|1,258.00|1,258.00|1,258.00|1,258.00|    1258|
    |   mean|   72.36|   72.84|   71.92|   72.39| 8222093|
    | stddev|    6.77|    6.77|    6.74|    6.76| 4519780|
    |    min|   56.39|   57.06|   56.30|   56.42| 2094900|
    |    max|   90.80|   90.97|   89.25|   90.47|80898100|
    +-------+--------+--------+--------+--------+--------+
    


#### Create a new dataframe with a column called HV Ratio that is the ratio of the High Price versus volume of stock traded for a day.


```python
df2 = df.withColumn("HV Ratio",df["High"]/df["Volume"])#.show()
# df2.show()
df2.select('HV Ratio').show()
```

    +--------------------+
    |            HV Ratio|
    +--------------------+
    |4.819714653321546E-6|
    |6.290848613094555E-6|
    |4.669412994783916E-6|
    |7.367338463826307E-6|
    |8.915604778943901E-6|
    |8.644477436914568E-6|
    |9.351828421515645E-6|
    | 8.29141562102703E-6|
    |7.712212102001476E-6|
    |7.071764823529412E-6|
    |1.015495466386981E-5|
    |6.576354146362592...|
    | 5.90145296180676E-6|
    |8.547679455011844E-6|
    |8.420709512685392E-6|
    |1.041448341728929...|
    |8.316075414862431E-6|
    |9.721183814992126E-6|
    |8.029436027707578E-6|
    |6.307432259386365E-6|
    +--------------------+
    only showing top 20 rows
    



```python
# What day had the Peak High in Price?
df.orderBy(df["High"].desc()).head(1)[0][0]
```




    datetime.datetime(2015, 1, 13, 0, 0)




```python
# what is the mean of the Close column?
from pyspark.sql.functions import mean
df.select(mean('Close')).show()
```

    +-----------------+
    |       avg(Close)|
    +-----------------+
    |72.38844998012726|
    +-----------------+
    



```python
# max and min of the Volumn column
from pyspark.sql.functions import max,min
df.select(max("Volume"),min("Volume")).show()
```

    +-----------+-----------+
    |max(Volume)|min(Volume)|
    +-----------+-----------+
    |   80898100|    2094900|
    +-----------+-----------+
    



```python
# How many days was the Close lower than 60 dollars?
df.filter("Close < 60").count()
```




    81




```python
# or 
df.filter(df['Close'] < 60).count()
```




    81




```python
# or 
from pyspark.sql.functions import count
result = df.filter(df['Close'] < 60)
result.select(count('Close')).show()
```

    +------------+
    |count(Close)|
    +------------+
    |          81|
    +------------+
    



```python
# (Number of Days High>80)/(Total Days in the datas
(df.filter(df["High"]>80).count()*1.0/df.count())*100
```




    9.141494435612083




```python
# Pearson correlation between high and volumn
from pyspark.sql.functions import corr
df.select(corr("High","Volume")).show()
```

    +-------------------+
    | corr(High, Volume)|
    +-------------------+
    |-0.3384326061737161|
    +-------------------+
    



```python
# the max High per year
from pyspark.sql.functions import year
yeardf = df.withColumn("Year", year(df["Date"]))
max_df = yeardf.groupBy("Year").max
max_df.select('Year','max(High)').show)
```
