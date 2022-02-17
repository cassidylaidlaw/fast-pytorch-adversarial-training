# Faster PyTorch adversarial training

This repository contains some information and code to help speed up typical adversarial training workflows in PyTorch. There are a few useful tricks that can give a 3-4x speedup, especially on newer NVidia GPUs with tensor cores:
 * Automatic mixed-precision during attacks
 * No calculation of parameter gradients during attacks
 * Fusion of conv and batchnorm layers into a single conv layer during attacks
 * Using NWHC layout

Say you have an adversarial training loop that looks something like this:
```python
model = MyModel()
model.cuda()
...
for inputs, labels in train_loader:
    ...
    model.eval()
    adv_inputs = attack(model, inputs, labels)
    model.train()
    loss = loss_fn(model(adv_inputs), labels)
    loss.backward()
    ...
```
If you add `fast_attack_model.py` into your project, you can modify your training loop as follows:
```python
from fast_attack_model import FastAttackModel
...
model = MyModel()
model.cuda()
fast_model = FastAttackModel(model)  # Create a FastAttackModel
...
for inputs, labels in train_loader:
    ...
    model.eval()
    fast_model.update()  # Update FastAttackModel with latest model parameters
    adv_inputs = attack(fast_model, inputs, labels)  # Call attack with fast_model
    model.train()
    loss = loss_fn(model(adv_inputs), labels)
    loss.backward()
    ...
```
This should generally give a significant speedup as each forward and backward pass through `fast_model` during the attack will be much faster than a forward and backward pass through `model`. However, there are some caveats and dangers:
 * In order to get the most speedup, make sure that all your layer sizes (number of channels for conv layers and number of inputs/outputs for linear layers) as well as your batch size are multiples of 8. See [here](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#troubleshooting) for more information. You should also make sure your batch sizes are large enough to saturate the GPU (you might need a bigger batch size with the faster model).
 * Because this module uses half-precision for calculating your model during attacks, numbers may underflow or overflow. Thus, you can get gradient values in your attack that are either 0 or infinity. If your gradient values are 0, one helpful trick is to scale the loss before calling backward in your adversarial attack. See [here](https://pytorch.org/docs/stable/amp.html#gradient-scaling) for more information about this trick. You may also want to filter out any infinite gradient values in your attack code before using the gradient to update the input.
