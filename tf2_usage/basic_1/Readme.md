# Keras 最基本的例子

点击查看 [示例代码](https://github.com/silenceli/learning/tree/master/tf2_usage/basic_1/main.py)，通过执行 [local_run 脚本](https://github.com/silenceli/learning/tree/master/tf2_usage/basic_1/local_run.sh) 可以运行之。

代码基于 `tensorflow 2.4`，使用 `keras 接口` 对 [mnist 数据集](http://yann.lecun.com/exdb/mnist/) 进行训练。

我们采用了最简单的模型：

`输入层 [batch_size, 784] ->  隐藏层1 (784, 256)  ->  隐藏层2 (256, 128)  ->  隐藏层3 (128, 10) -> 输出层 softmax [batch_size, 10]`

- 模型的输入 shape 为 `[batch_size, 784]`
- 隐藏层 1 的参数量为：`784 * 256`
- 模型的输出 shape 为 `[batch_size, 10]`

最终可以看到执行效果：

```
2022-10-13 11:36:22.141198: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
2022-10-13 11:36:22.141284: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
2022-10-13 11:36:22.141301: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 256)               200960    
_________________________________________________________________
dense_1 (Dense)              (None, 128)               32896     
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1290      
_________________________________________________________________
softmax (Softmax)            (None, 10)                0         
=================================================================
Total params: 235,146
Trainable params: 235,146
Non-trainable params: 0
_________________________________________________________________
Epoch 1/5
2022-10-13 11:36:23.817252: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2022-10-13 11:36:23.819401: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 2400000000 Hz
3000/3000 [==============================] - 6s 2ms/step - loss: 3.9178 - accuracy: 0.8433 - val_loss: 0.2660 - val_accuracy: 0.9282
Epoch 2/5
3000/3000 [==============================] - 6s 2ms/step - loss: 0.2671 - accuracy: 0.9311 - val_loss: 0.2645 - val_accuracy: 0.9327
Epoch 3/5
3000/3000 [==============================] - 6s 2ms/step - loss: 0.2667 - accuracy: 0.9338 - val_loss: 0.2344 - val_accuracy: 0.9484
Epoch 4/5
3000/3000 [==============================] - 6s 2ms/step - loss: 0.2465 - accuracy: 0.9436 - val_loss: 0.2580 - val_accuracy: 0.9524
Epoch 5/5
3000/3000 [==============================] - 6s 2ms/step - loss: 0.2429 - accuracy: 0.9456 - val_loss: 0.2435 - val_accuracy: 0.9438
2022-10-13 11:36:54.573112: W tensorflow/python/util/util.cc:348] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
INFO:tensorflow:Assets written to: /tmp/model/assets
I1013 11:36:54.993173 139895599028032 builder_impl.py:774] Assets written to: /tmp/model/assets
```
