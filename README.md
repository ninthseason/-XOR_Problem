### 用前馈神经网络解决XOR问题

激活函数选用 Leaky_RELU，很难训练失败。经实验 RELU 和 Sigmoid 都不稳定。

可以不 uniform 参数，似乎对效果影响不大。


如果非要用 RELU，可以通过合适的初始化参数提高稳定性。
比如三层线性层参数全部初始化为：
权重 uniform(-1, 1)
偏置 constant(0.5)
