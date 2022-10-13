import tensorflow as tf
import numpy as np

from tensorflow.compat.v1 import app

# tf.app.run()：https://blog.csdn.net/helei001/article/details/51859423

flags = app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_integer("epochs", 2, "epochs")
flags.DEFINE_string("model_dir", "/tmp/model/", "model dir")


def run(argv=None):
    """
    (Pdb) type(x)
    <class 'numpy.ndarray'>
    
    (Pdb) x.shape
    (60000, 28, 28)
    (Pdb) y.shape
    (60000,)
    (Pdb) x_val.shape
    (10000, 28, 28)
    (Pdb) y_val.shape
    (10000,)
    """
    # 加载数据
    (x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

    ## 组织数据成为 datasets
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    x = tf.reshape(x, [-1, 784])
    x_val = tf.convert_to_tensor(x_val, dtype=tf.float64)
    x_val = tf.reshape(x_val, [-1, 784])
    
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    y = tf.cast(y, dtype=tf.float64)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)    
    y_val = tf.one_hot(y_val, depth=10)
    y_val = tf.cast(y_val, dtype=tf.float64)

    dataset_train = tf.data.Dataset.from_tensor_slices((x,y))
    dataset_val = tf.data.Dataset.from_tensor_slices((x_val,y_val))

    dataset_train = dataset_train.batch(20).shuffle(buffer_size=100)
    dataset_val = dataset_val.batch(20).shuffle(buffer_size=100)

    # 构建模型
    """
    (batch_size, 784) ->  (784, 256) -> (256, 128) -> (128, 10)
    """

    network = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax(axis=-1)
    ])

    # -1 非法，None 可以
    network.build(input_shape=(None, 784))

    # 模型装配
    network.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])
    
    network.summary()

    network.fit(
        x=dataset_train,
        epochs=FLAGS.epochs,
        validation_data=dataset_val)

    tf.saved_model.save(network, FLAGS.model_dir)


if __name__ == "__main__":
    app.run(run)