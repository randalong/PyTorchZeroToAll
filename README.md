# PyTorchZeroToAll

This repository implements example codes introduced in the ***PyTorchZeroToAll*** course offered by ***HKUST***. 

## 1. Version check

Run ```print(torch.__version__)``` to check version status.

You can find the codes in [version_check.py].

[version_check.py]: https://github.com/Tom-Pomelo/PyTorchZeroToAll/blob/master/1_version_check.py

## 2. Linear model

Assume a linear correlation between datasets `x` and `y`, namely, <center><a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;$$\bar&space;y&space;=&space;w&space;*&space;x$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_cm&space;$$\bar&space;y&space;=&space;w&space;*&space;x$$" title="$$\bar y = w * x$$" /></a></center>

Define a `loss function`, namely, <center><a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;$$loss&space;=&space;(\bar&space;y&space;-&space;y)^2&space;=&space;(w&space;*&space;x&space;-&space;y)^2&space;=&space;\frac{1}{N}\sum_{n&space;=&space;1}^N(\bar&space;y_n&space;-&space;y_n)^2$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_cm&space;$$loss&space;=&space;(\bar&space;y&space;-&space;y)^2&space;=&space;(w&space;*&space;x&space;-&space;y)^2&space;=&space;\frac{1}{N}\sum_{n&space;=&space;1}^N(\bar&space;y_n&space;-&space;y_n)^2$$" title="$$loss = (\bar y - y)^2 = (w * x - y)^2 = \frac{1}{N}\sum_{n = 1}^N(\bar y_n - y_n)^2$$" /></a></center>

You can find the codes in [linear_model.py] to plot `loss function` versus `w`, 

[linear_model.py]: https://github.com/Tom-Pomelo/PyTorchZeroToAll/blob/master/2_linear_model.py

## 3. Gradient model

After retrieving the plot between `loss function` and `w`, we try to minimize the `loss function`.

That is, we need to walk along the path where 

<center>
<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;$$w&space;=&space;w&space;-&space;\alpha\frac{\partial{loss}}{\partial{w}}$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_cm&space;$$w&space;=&space;w&space;-&space;\alpha\frac{\partial{loss}}{\partial{w}}$$" title="$$w = w - \alpha\frac{\partial{loss}}{\partial{w}}$$" /></a>.
</center>

Such a path is formally called ***Gradient Descent***.

You can find the codes in [gradient_model.py] to show that `loss function` decreases as walking along the above path.

[gradient_model.py]: https://github.com/Tom-Pomelo/PyTorchZeroToAll/blob/master/3_gradient_model.py