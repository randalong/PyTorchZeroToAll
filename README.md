# PyTorchZeroToAll

This repository implements example codes introduced in the ***PyTorchZeroToAll*** course offered by ***HKUST***. 

## 1. Version check

Run ```print(torch.__version__)``` to check version status.

You can find the codes in [version_check.py].

[version_check.py]: https://github.com/Tom-Pomelo/PyTorchZeroToAll/blob/master/1_version_check.py

## 2. Linear model

Assume a linear correlation between datasets `x` and `y`, namely, $\bar{y} = w * x$.

Define a `loss function`, namely, $$loss = (\bar y - y)^2 = (w * x - y)^2 = \frac{1}{N}\sum_{n = 1}^N(\bar y_n - y_n)^2$$.

You can find the codes in [linear_model.py] to plot `loss function` versus `w`, 

[linear_model.py]: https://github.com/Tom-Pomelo/PyTorchZeroToAll/blob/master/2_linear_model.py

## 3. Gradient model

After retrieving the plot between `loss function` and `w`, we try to minimize the `loss function`.

That is, we need to walk along the path where $$w = w - \alpha\frac{\partial{loss}}{\partial{w}}$$.

Such a path is formally called ***Gradient Descent***.

You can find the codes in [gradient_model.py] to show that `loss function` decreases as walking along the above path.

[gradient_model.py]: https://github.com/Tom-Pomelo/PyTorchZeroToAll/blob/master/3_gradient_model.py