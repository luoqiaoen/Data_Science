

```python
import pandas as pd
df = pd.read_csv('iris.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns = ['sepal_length','sepal_width','petal_length','petal_width','target']
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df.drop('target',axis=1)
y = df['target'].apply(int)
```

## Train Test Split


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

# Estimators


```python
import tensorflow as tf
```

## Feature Column


```python
X.columns
```




    Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], dtype='object')




```python
feat_cols = []

for col in X.columns:
    feat_cols.append(tf.feature_column.numeric_column(col))
```


```python
feat_cols
```




    [_NumericColumn(key='sepal_length', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
     _NumericColumn(key='sepal_width', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
     _NumericColumn(key='petal_length', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
     _NumericColumn(key='petal_width', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)]



## Input Function


```python
# there is also a pandas_input_fn we'll see in the exercise!!
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=5,shuffle=True)
```


```python
classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3,feature_columns=feat_cols)
```

    INFO:tensorflow:Using default config.
    WARNING:tensorflow:Using temporary folder as model directory: /var/folders/gr/bxc92_ln64b6m98lgkjnjcn40000gn/T/tmph4g2h5_r
    INFO:tensorflow:Using config: {'_model_dir': '/var/folders/gr/bxc92_ln64b6m98lgkjnjcn40000gn/T/tmph4g2h5_r', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
    graph_options {
      rewrite_options {
        meta_optimizer_iterations: ONE
      }
    }
    , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x1a39981a90>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}



```python
classifier.train(input_fn=input_func,steps=50)
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from /var/folders/gr/bxc92_ln64b6m98lgkjnjcn40000gn/T/tmph4g2h5_r/model.ckpt-50
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 50 into /var/folders/gr/bxc92_ln64b6m98lgkjnjcn40000gn/T/tmph4g2h5_r/model.ckpt.
    INFO:tensorflow:loss = 2.8195987, step = 51
    INFO:tensorflow:Saving checkpoints for 100 into /var/folders/gr/bxc92_ln64b6m98lgkjnjcn40000gn/T/tmph4g2h5_r/model.ckpt.
    INFO:tensorflow:Loss for final step: 2.3176205.





    <tensorflow.python.estimator.canned.dnn.DNNClassifier at 0x1a39981780>



# Model Evaluation


```python
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
```


```python
note_predictions = list(classifier.predict(input_fn=pred_fn))
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from /var/folders/gr/bxc92_ln64b6m98lgkjnjcn40000gn/T/tmph4g2h5_r/model.ckpt-100
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.



```python
note_predictions[0]
```




    {'logits': array([-4.1635566,  4.1832147,  5.891665 ], dtype=float32),
     'probabilities': array([3.6370835e-05, 1.5335923e-01, 8.4660435e-01], dtype=float32),
     'class_ids': array([2]),
     'classes': array([b'2'], dtype=object)}




```python
final_preds  = []
for pred in note_predictions:
    final_preds.append(pred['class_ids'][0])
```


```python
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,final_preds))
```

    [[15  0  0]
     [ 0 13  3]
     [ 0  0 14]]



```python
print(classification_report(y_test,final_preds))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        15
               1       1.00      0.81      0.90        16
               2       0.82      1.00      0.90        14
    
       micro avg       0.93      0.93      0.93        45
       macro avg       0.94      0.94      0.93        45
    weighted avg       0.95      0.93      0.93        45
    



```python

```
