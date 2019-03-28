
# Linear Regression

Refer to http://spark.apache.org/docs/latest/ml-features.html for information about how to transform data. 


```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('lr_example').getOrCreate()
```


```python
from pyspark.ml.regression import LinearRegression
```


```python
data = spark.read.csv("Ecommerce_Customers.csv",inferSchema=True,header=True)
```


```python
data.printSchema()
```

    root
     |-- Email: string (nullable = true)
     |-- Address: string (nullable = true)
     |-- Avatar: string (nullable = true)
     |-- Avg Session Length: double (nullable = true)
     |-- Time on App: double (nullable = true)
     |-- Time on Website: double (nullable = true)
     |-- Length of Membership: double (nullable = true)
     |-- Yearly Amount Spent: double (nullable = true)
    



```python
data.show()
```

    +--------------------+--------------------+----------------+------------------+------------------+------------------+--------------------+-------------------+
    |               Email|             Address|          Avatar|Avg Session Length|       Time on App|   Time on Website|Length of Membership|Yearly Amount Spent|
    +--------------------+--------------------+----------------+------------------+------------------+------------------+--------------------+-------------------+
    |mstephenson@ferna...|835 Frank TunnelW...|          Violet| 34.49726772511229| 12.65565114916675| 39.57766801952616|  4.0826206329529615|  587.9510539684005|
    |   hduke@hotmail.com|4547 Archer Commo...|       DarkGreen| 31.92627202636016|11.109460728682564|37.268958868297744|    2.66403418213262|  392.2049334443264|
    |    pallen@yahoo.com|24645 Valerie Uni...|          Bisque|33.000914755642675|11.330278057777512|37.110597442120856|   4.104543202376424| 487.54750486747207|
    |riverarebecca@gma...|1414 David Throug...|     SaddleBrown| 34.30555662975554|13.717513665142507| 36.72128267790313|   3.120178782748092|  581.8523440352177|
    |mstephens@davidso...|14023 Rodriguez P...|MediumAquaMarine| 33.33067252364639|12.795188551078114| 37.53665330059473|   4.446308318351434|  599.4060920457634|
    |alvareznancy@luca...|645 Martha Park A...|     FloralWhite|33.871037879341976|12.026925339755056| 34.47687762925054|   5.493507201364199|   637.102447915074|
    |katherine20@yahoo...|68388 Reyes Light...|   DarkSlateBlue| 32.02159550138701|11.366348309710526| 36.68377615286961|   4.685017246570912|  521.5721747578274|
    |  awatkins@yahoo.com|Unit 6538 Box 898...|            Aqua|32.739142938380326| 12.35195897300293| 37.37335885854755|  4.4342734348999375|  549.9041461052942|
    |vchurch@walter-ma...|860 Lee KeyWest D...|          Salmon| 33.98777289568564|13.386235275676436|37.534497341555735|  3.2734335777477144|  570.2004089636196|
    |    bonnie69@lin.biz|PSC 2734, Box 525...|           Brown|31.936548618448917|11.814128294972196| 37.14516822352819|   3.202806071553459|  427.1993848953282|
    |andrew06@peterson...|26104 Alexander G...|          Tomato|33.992572774953736|13.338975447662113| 37.22580613162114|   2.482607770510596|  492.6060127179966|
    |ryanwerner@freema...|Unit 2413 Box 034...|          Tomato| 33.87936082480498|11.584782999535266| 37.08792607098381|    3.71320920294043|  522.3374046069357|
    |   knelson@gmail.com|6705 Miller Orcha...|       RoyalBlue|29.532428967057943|10.961298400154098| 37.42021557502538|   4.046423164299585|  408.6403510726275|
    |wrightpeter@yahoo...|05302 Dunlap Ferr...|          Bisque| 33.19033404372265|12.959226091609382|36.144666700041924|   3.918541839158999|  573.4158673313865|
    |taylormason@gmail...|7773 Powell Sprin...|        DarkBlue|32.387975853153876|13.148725692056516| 36.61995708279922|   2.494543646659249|  470.4527333009554|
    | jstark@anderson.com|49558 Ramirez Roa...|            Peru|30.737720372628182|12.636606052000127|36.213763093698624|  3.3578468423262944|  461.7807421962299|
    | wjennings@gmail.com|6362 Wilson Mount...|      PowderBlue| 32.12538689728784|11.733861690857394|  34.8940927514398|  3.1361327164897803| 457.84769594494855|
    |rebecca45@hale-ba...|8982 Burton RowWi...|       OliveDrab|32.338899323067196|12.013194694014402| 38.38513659413844|   2.420806160901484| 407.70454754954415|
    |alejandro75@hotma...|64475 Andre Club ...|            Cyan|32.187812045932155|  14.7153875441565| 38.24411459434352|   1.516575580831944|  452.3156754800354|
    |samuel46@love-wes...|544 Alexander Hei...|   LightSeaGreen| 32.61785606282345|13.989592555825254|37.190503800397956|   4.064548550437977|   605.061038804892|
    +--------------------+--------------------+----------------+------------------+------------------+------------------+--------------------+-------------------+
    only showing top 20 rows
    



```python
data.head(2)
```




    [Row(Email='mstephenson@fernandez.com', Address='835 Frank TunnelWrightmouth, MI 82180-9605', Avatar='Violet', Avg Session Length=34.49726772511229, Time on App=12.65565114916675, Time on Website=39.57766801952616, Length of Membership=4.0826206329529615, Yearly Amount Spent=587.9510539684005),
     Row(Email='hduke@hotmail.com', Address='4547 Archer CommonDiazchester, CA 06566-8576', Avatar='DarkGreen', Avg Session Length=31.92627202636016, Time on App=11.109460728682564, Time on Website=37.268958868297744, Length of Membership=2.66403418213262, Yearly Amount Spent=392.2049334443264)]




```python
for item in data.head():
    print(item)
```

    mstephenson@fernandez.com
    835 Frank TunnelWrightmouth, MI 82180-9605
    Violet
    34.49726772511229
    12.65565114916675
    39.57766801952616
    4.0826206329529615
    587.9510539684005


## Setting Up DataFrame for Machine Learning 
two columns needed, ("label", "feature")


```python
# Import VectorAssembler and Vectors
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

data.columns
```




    ['Email',
     'Address',
     'Avatar',
     'Avg Session Length',
     'Time on App',
     'Time on Website',
     'Length of Membership',
     'Yearly Amount Spent']




```python
assembler = VectorAssembler(
    inputCols=["Avg Session Length", "Time on App", 
               "Time on Website","Length of Membership"],
    outputCol="features") 
# features are "Avg Session Length", "Time on App", "Time on Website",'Length of Membership'
output = assembler.transform(data)
output.select("features").head(1)
```




    [Row(features=DenseVector([34.4973, 12.6557, 39.5777, 4.0826]))]




```python
output.show()
```

    +--------------------+--------------------+----------------+------------------+------------------+------------------+--------------------+-------------------+--------------------+
    |               Email|             Address|          Avatar|Avg Session Length|       Time on App|   Time on Website|Length of Membership|Yearly Amount Spent|            features|
    +--------------------+--------------------+----------------+------------------+------------------+------------------+--------------------+-------------------+--------------------+
    |mstephenson@ferna...|835 Frank TunnelW...|          Violet| 34.49726772511229| 12.65565114916675| 39.57766801952616|  4.0826206329529615|  587.9510539684005|[34.4972677251122...|
    |   hduke@hotmail.com|4547 Archer Commo...|       DarkGreen| 31.92627202636016|11.109460728682564|37.268958868297744|    2.66403418213262|  392.2049334443264|[31.9262720263601...|
    |    pallen@yahoo.com|24645 Valerie Uni...|          Bisque|33.000914755642675|11.330278057777512|37.110597442120856|   4.104543202376424| 487.54750486747207|[33.0009147556426...|
    |riverarebecca@gma...|1414 David Throug...|     SaddleBrown| 34.30555662975554|13.717513665142507| 36.72128267790313|   3.120178782748092|  581.8523440352177|[34.3055566297555...|
    |mstephens@davidso...|14023 Rodriguez P...|MediumAquaMarine| 33.33067252364639|12.795188551078114| 37.53665330059473|   4.446308318351434|  599.4060920457634|[33.3306725236463...|
    |alvareznancy@luca...|645 Martha Park A...|     FloralWhite|33.871037879341976|12.026925339755056| 34.47687762925054|   5.493507201364199|   637.102447915074|[33.8710378793419...|
    |katherine20@yahoo...|68388 Reyes Light...|   DarkSlateBlue| 32.02159550138701|11.366348309710526| 36.68377615286961|   4.685017246570912|  521.5721747578274|[32.0215955013870...|
    |  awatkins@yahoo.com|Unit 6538 Box 898...|            Aqua|32.739142938380326| 12.35195897300293| 37.37335885854755|  4.4342734348999375|  549.9041461052942|[32.7391429383803...|
    |vchurch@walter-ma...|860 Lee KeyWest D...|          Salmon| 33.98777289568564|13.386235275676436|37.534497341555735|  3.2734335777477144|  570.2004089636196|[33.9877728956856...|
    |    bonnie69@lin.biz|PSC 2734, Box 525...|           Brown|31.936548618448917|11.814128294972196| 37.14516822352819|   3.202806071553459|  427.1993848953282|[31.9365486184489...|
    |andrew06@peterson...|26104 Alexander G...|          Tomato|33.992572774953736|13.338975447662113| 37.22580613162114|   2.482607770510596|  492.6060127179966|[33.9925727749537...|
    |ryanwerner@freema...|Unit 2413 Box 034...|          Tomato| 33.87936082480498|11.584782999535266| 37.08792607098381|    3.71320920294043|  522.3374046069357|[33.8793608248049...|
    |   knelson@gmail.com|6705 Miller Orcha...|       RoyalBlue|29.532428967057943|10.961298400154098| 37.42021557502538|   4.046423164299585|  408.6403510726275|[29.5324289670579...|
    |wrightpeter@yahoo...|05302 Dunlap Ferr...|          Bisque| 33.19033404372265|12.959226091609382|36.144666700041924|   3.918541839158999|  573.4158673313865|[33.1903340437226...|
    |taylormason@gmail...|7773 Powell Sprin...|        DarkBlue|32.387975853153876|13.148725692056516| 36.61995708279922|   2.494543646659249|  470.4527333009554|[32.3879758531538...|
    | jstark@anderson.com|49558 Ramirez Roa...|            Peru|30.737720372628182|12.636606052000127|36.213763093698624|  3.3578468423262944|  461.7807421962299|[30.7377203726281...|
    | wjennings@gmail.com|6362 Wilson Mount...|      PowderBlue| 32.12538689728784|11.733861690857394|  34.8940927514398|  3.1361327164897803| 457.84769594494855|[32.1253868972878...|
    |rebecca45@hale-ba...|8982 Burton RowWi...|       OliveDrab|32.338899323067196|12.013194694014402| 38.38513659413844|   2.420806160901484| 407.70454754954415|[32.3388993230671...|
    |alejandro75@hotma...|64475 Andre Club ...|            Cyan|32.187812045932155|  14.7153875441565| 38.24411459434352|   1.516575580831944|  452.3156754800354|[32.1878120459321...|
    |samuel46@love-wes...|544 Alexander Hei...|   LightSeaGreen| 32.61785606282345|13.989592555825254|37.190503800397956|   4.064548550437977|   605.061038804892|[32.6178560628234...|
    +--------------------+--------------------+----------------+------------------+------------------+------------------+--------------------+-------------------+--------------------+
    only showing top 20 rows
    


## Train Test Split


```python
final_data = output.select("features",'Yearly Amount Spent')
train_data,test_data = final_data.randomSplit([0.7,0.3])
train_data.describe().show()
```

    +-------+-------------------+
    |summary|Yearly Amount Spent|
    +-------+-------------------+
    |  count|                356|
    |   mean|  501.2961051169299|
    | stddev|  81.06698968643927|
    |    min| 256.67058229005585|
    |    max|  765.5184619388373|
    +-------+-------------------+
    



```python
test_data.describe().show()
```

    +-------+-------------------+
    |summary|Yearly Amount Spent|
    +-------+-------------------+
    |  count|                144|
    |   mean| 494.41392852547426|
    | stddev|  74.85468949078356|
    |    min|  319.9288698031936|
    |    max|  725.5848140556806|
    +-------+-------------------+
    


## Fit Linear Regression Model


```python
lr = LinearRegression(labelCol='Yearly Amount Spent')
lrModel = lr.fit(train_data)
print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,lrModel.intercept))
```

    Coefficients: [25.92302306401339,38.4497785743354,0.565556565310133,61.43970203843536] Intercept: -1059.124708337252



```python
test_results = lrModel.evaluate(test_data)
```


```python
test_results.residuals.show()
```

    +-------------------+
    |          residuals|
    +-------------------+
    |  -11.2193767412856|
    |  11.43007079268034|
    | -2.948644265438986|
    |  7.458807105067308|
    |  4.506377170359087|
    | -5.372941674344759|
    | 3.9764871155747414|
    | 0.5545083510176028|
    |-3.2087750434934605|
    | -8.915117379572791|
    | -16.71209178844697|
    |-13.522676893995822|
    | 18.534765531418543|
    | 17.611002220845023|
    |-0.7250406470051303|
    |  2.764175508806204|
    | -4.323604408323774|
    |  -9.01349803781153|
    |  8.796404639668992|
    | -2.117435066688813|
    +-------------------+
    only showing top 20 rows
    



```python
test_results.r2
```




    0.9828184315563212




```python
unlabeled_data = test_data.select('features')
predictions = lrModel.transform(unlabeled_data)
predictions.show()
```

    +--------------------+------------------+
    |            features|        prediction|
    +--------------------+------------------+
    |[30.3931845423455...| 331.1482465444792|
    |[30.7377203726281...|450.35067140354954|
    |[30.8794843441274...|493.15524425029366|
    |[30.9716756438877...| 487.1798026518254|
    |[31.0472221394875...| 387.9910220186623|
    |[31.0613251567161...|492.92839973224636|
    |[31.3662121671876...| 426.6123954409102|
    |[31.3895854806643...| 409.5151027089653|
    |[31.4252268808548...| 533.9754936982554|
    |[31.5261978982398...| 418.0096435719106|
    |[31.5702008293202...| 562.6575839298519|
    |[31.5741380228732...| 557.9319490545827|
    |[31.6005122003032...| 460.6380859596784|
    |[31.6098395733896...|426.93454743026314|
    |[31.6610498227460...|417.08339422690597|
    |[31.7366356860502...| 494.1692707467257|
    |[31.7656188210424...| 500.8776860439309|
    |[31.8279790554652...|449.01624558475305|
    |[31.8512531286083...| 464.1958420271294|
    |[31.8530748017465...| 461.4025585290408|
    +--------------------+------------------+
    only showing top 20 rows
    



```python
print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))
```

    RMSE: 9.777710829809825
    MSE: 95.60362907138034

