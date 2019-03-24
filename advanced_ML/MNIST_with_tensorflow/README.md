

```python
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

tf.logging.set_verbosity(old_v)
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz



```python
type(mnist)
```




    tensorflow.contrib.learn.python.learn.datasets.base.Datasets




```python
mnist.train.images
```




    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)




```python
mnist.train.num_examples
```




    55000




```python
mnist.test.num_examples
```




    10000




```python
mnist.validation.num_examples
```




    5000



## Visualization


```python
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
mnist.train.images[1].shape
```




    (784,)




```python
plt.imshow(mnist.train.images[1].reshape(28,28))
```




    <matplotlib.image.AxesImage at 0xb477d7f28>




![png](Basic_MNIST_with_TF_files/Basic_MNIST_with_TF_9_1.png)



```python
plt.imshow(mnist.train.images[1].reshape(28,28),cmap='gist_gray')
```




    <matplotlib.image.AxesImage at 0xb478bd1d0>




![png](Basic_MNIST_with_TF_files/Basic_MNIST_with_TF_10_1.png)



```python
mnist.train.images[1].max()
```




    1.0




```python
plt.imshow(mnist.train.images[1].reshape(784,1),cmap='gist_gray',aspect=0.02)
```




    <matplotlib.image.AxesImage at 0xb479f7400>




![png](Basic_MNIST_with_TF_files/Basic_MNIST_with_TF_12_1.png)


## Create the Model


```python
x = tf.placeholder(tf.float32,shape=[None,784])
```


```python
W = tf.Variable(tf.zeros([784,10]))
```


```python
b = tf.Variable(tf.zeros([10]))
```


```python
# Create the Graph
y = tf.matmul(x,W) + b 
```

### loss and optimizer


```python
y_true = tf.placeholder(tf.float32,[None,10])
```

### Cross Entropy


```python
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)
```

## Create Session


```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    # Train the model for 1000 steps on the training set
    # Using built in batch feeder from mnist for convenience
    
    for step in range(1000):
        
        batch_x , batch_y = mnist.train.next_batch(100)
        
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y})
        
    # Test the Train Model
    matches = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
    
    acc = tf.reduce_mean(tf.cast(matches,tf.float32))
    
    print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))
```


```python

```
