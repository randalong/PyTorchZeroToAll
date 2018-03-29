# PyTorchZeroToAll

This repository implements example codes introduced in the ***PyTorchZeroToAll*** course offered by ***HKUST***. 
___
## Catalog

* [1. Version check](#version_check)
* [2. Linear model](#linear_model)
* [3. Gradient model](#gradient_model)
* [4. Backward propagation](#backward)
* [5. Linear regression in PyTorch](linear_regression)
* [6. Logistic regression](logistic_regression)

___
## <a name = "version_check" /> 1. Version check

Run ```print(torch.__version__)``` to check version status.

You can find the codes in [version_check.py].

[version_check.py]: https://github.com/Tom-Pomelo/PyTorchZeroToAll/blob/master/1_version_check.py

## <a name = "linear_model" /> 2. Linear model

Assume a linear correlation between datasets `x` and `y`, namely, <center><a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;$$\bar&space;y&space;=&space;w&space;*&space;x$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_cm&space;$$\bar&space;y&space;=&space;w&space;*&space;x$$" title="$$\bar y = w * x$$" /></a></center>

Define a `loss function`, namely, <center><a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;$$loss&space;=&space;(\bar&space;y&space;-&space;y)^2&space;=&space;(w&space;*&space;x&space;-&space;y)^2&space;=&space;\frac{1}{N}\sum_{n&space;=&space;1}^N(\bar&space;y_n&space;-&space;y_n)^2$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_cm&space;$$loss&space;=&space;(\bar&space;y&space;-&space;y)^2&space;=&space;(w&space;*&space;x&space;-&space;y)^2&space;=&space;\frac{1}{N}\sum_{n&space;=&space;1}^N(\bar&space;y_n&space;-&space;y_n)^2$$" title="$$loss = (\bar y - y)^2 = (w * x - y)^2 = \frac{1}{N}\sum_{n = 1}^N(\bar y_n - y_n)^2$$" /></a></center>

You can find the codes in [linear_model.py] to plot `loss function` versus `w`, 

[linear_model.py]: https://github.com/Tom-Pomelo/PyTorchZeroToAll/blob/master/2_linear_model.py

## <a name = "gradient_model" /> 3. Gradient model

After retrieving the plot between `loss function` and `w`, we try to minimize the `loss function`.

That is, we need to walk along the path where 

<center>
<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;$$w&space;=&space;w&space;-&space;\alpha\frac{\partial{loss}}{\partial{w}}$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_cm&space;$$w&space;=&space;w&space;-&space;\alpha\frac{\partial{loss}}{\partial{w}}$$" title="$$w = w - \alpha\frac{\partial{loss}}{\partial{w}}$$" /></a>.
</center>

Such a path is formally called ***Gradient Descent***.

You can find the codes in [gradient_model.py] to show that `loss function` decreases as walking along the above path.

## <a name = "backward" /> 4. Backward propagation

`PyTorch` is convenient in that it enables automatic gradient calculation. 

Eg: `w = Variable(torch.Tensor([1.0]), requires_grad=True)`

Rewrite [gradient_model.py] with PyTorch using backward propagation.

You can find the codes in [backward_propagation.py] with `PyTorch` style.

## <a name = "linear_regression" /> 5. Linear regression in PyTorch

This is the first time we have to construct a model in `PyTorch`.

```
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.Linear(x)
        return y_pred
```

Notice that `_init_` and `forward` functions must be implemented. 

After the model class is constructed, we have to construct `loss function` and `optimizer`. 

```
criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
```

You can find the codes in [linear_regression.py] to see how a model is constructed and you may would like to try for different `loss functions` and `optimizers`.

## <a name = "logistic_regression" /> 6. Logistic regression

Have you ever hesitated whether or not you should propose to your dream lover?

This is called `binary prediction`.

A very important function you may want to know: [Sigmoid]

Back to our example, we now have datasets `x` and `y`.

```
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]]))
y_data = Variable(torch.Tensor([[0.], [0.], [1.], [1.]]))
```

Obviously, `x` and `y` form a binary prediction relation.

You can find the codes in [logistic_regression.py].

### CopyRight All Rights Reserved

[logistic_regression.py]: https://github.com/Tom-Pomelo/PyTorchZeroToAll/blob/master/6_logistic_regression.py

[Sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function

[linear_regression.py]: https://github.com/Tom-Pomelo/PyTorchZeroToAll/blob/master/5_linear_regression.py

[backward_propagation.py]: https://github.com/Tom-Pomelo/PyTorchZeroToAll/blob/master/4_backward_propagation.py

[gradient_model.py]: https://github.com/Tom-Pomelo/PyTorchZeroToAll/blob/master/3_gradient_model.py