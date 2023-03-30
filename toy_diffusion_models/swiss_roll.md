```python
data, order = datasets.make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
data /= data.max()
doc.print(data.shape)
```

```
(1000, 3)
```
```python
plt.scatter(data[:, 0], data[:, 2], c=order, s=50, cmap='viridis')
doc.savefig(f'{Path(__file__).stem}/swiss_roll.png', title=f'Swiss Roll', close=True)
```

<table><tr><th>Swiss Roll</th></tr><tr><td><img style="align-self:center;" src="swiss_roll/swiss_roll.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/></td></tr></table>
```python
def forward_process(x, beta=0.1):
    return np.sqrt(1 - beta) * x + torch.normal(0, beta, size=x.shape)
```



```python
x = x_0
beta_t = 0
for step in range(Params.steps):
    if step % 5 == 0:
        row = t.figure_row()

    beta_t = np.sqrt(beta_t ** 2 + Params.beta ** 2)
    x = forward_process(x, beta=Params.beta)

    logger.store_metrics(beta_t=beta_t, silent=True)
    plt.scatter(x[:, 0], x[:, 2], c=order, s=50, cmap='viridis')
    row.savefig(f'{Path(__file__).stem}/diffusion_{step:03d}.png', title=f'Step {step}', close=True)
```

| **Step 0** | **Step 1** | **Step 2** | **Step 3** | **Step 4** |
|:----------:|:----------:|:----------:|:----------:|:----------:|
| <img style="align-self:center;" src="swiss_roll/diffusion_000.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_001.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_002.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_003.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_004.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> |
| **Step 5** | **Step 6** | **Step 7** | **Step 8** | **Step 9** |
| <img style="align-self:center;" src="swiss_roll/diffusion_005.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_006.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_007.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_008.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_009.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> |
| **Step 10** | **Step 11** | **Step 12** | **Step 13** | **Step 14** |
| <img style="align-self:center;" src="swiss_roll/diffusion_010.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_011.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_012.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_013.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_014.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> |
| **Step 15** | **Step 16** | **Step 17** | **Step 18** | **Step 19** |
| <img style="align-self:center;" src="swiss_roll/diffusion_015.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_016.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_017.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_018.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_019.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> |
| **Step 20** | **Step 21** | **Step 22** | **Step 23** | **Step 24** |
| <img style="align-self:center;" src="swiss_roll/diffusion_020.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_021.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_022.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_023.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_024.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> |
| **Step 25** | **Step 26** | **Step 27** | **Step 28** | **Step 29** |
| <img style="align-self:center;" src="swiss_roll/diffusion_025.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_026.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_027.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_028.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_029.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> |
| **Step 30** | **Step 31** | **Step 32** | **Step 33** | **Step 34** |
| <img style="align-self:center;" src="swiss_roll/diffusion_030.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_031.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_032.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_033.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_034.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> |
| **Step 35** | **Step 36** | **Step 37** | **Step 38** | **Step 39** |
| <img style="align-self:center;" src="swiss_roll/diffusion_035.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_036.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_037.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_038.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/diffusion_039.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> |
```python
plt.figure()
plt.plot(logger.summary_cache['beta_t'])
plt.ylim(0., 0.4)
doc.savefig(f'{Path(__file__).stem}/beta_t.png', title=f'Beta t', close=True)
```

<table><tr><th>Beta t</th></tr><tr><td><img style="align-self:center;" src="swiss_roll/beta_t.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/></td></tr></table>
```python
net = Diff2D(steps=Params.steps)


def backward_process(x, step, step_size=0.1):
    return net(x, step) * step_size + x


def train(x_0, optim, n_epochs=1000):

    for epoch in trange(n_epochs, desc='Epoch'):
        x = x_0
        for step in range(Params.steps):
            x_prime = forward_process(x, beta=Params.beta)
            loss = F.mse_loss(backward_process(x_prime, step=step), x)
            optim.zero_grad()
            loss.backward()
            optim.step()

            x = x_prime

            logger.store_metrics(loss=loss.item())

        logger.log_metrics_summary(silent=True, key_values={'epoch': epoch})


# logger.configure("diffusion-models", logger.now(f"%Y/%m-%d/{Path(__file__).stem}/%H%M%S.%f"))
optim = torch.optim.Adam(net.parameters(), lr=0.01)
train(x_0, optim, n_epochs=100)

losses, epochs = logger.read_metrics('loss/mean@mean', x_key="epoch@min")

plt.plot(epochs.to_numpy(), losses.to_numpy())
plt.ylim(0., 0.4)
doc.savefig(f'{Path(__file__).stem}/loss.png', title=f'Loss', close=True)
```

<table><tr><th>Loss</th></tr><tr><td><img style="align-self:center;" src="swiss_roll/loss.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/></td></tr></table>

The model is able to recover the original data.

```python
x_t = x_0 = torch.normal(0, 1, size=(1000, 3))
xs = [x_0.cpu().numpy()]
with torch.no_grad():
    for i, step in enumerate(tqdm(list(range(Params.steps))[::-1], leave=False)):
        if i % 5 == 0:
            row = t.figure_row()

        # for i in trange(100, desc="optim step"):
        x_t = backward_process(x_t, step=step, step_size=Params.beta)

        # xs.append(x_t.cpu().numpy())

        plt.scatter(x_t[:, 0], x_t[:, 2], s=50, cmap='viridis')
        row.savefig(f'{Path(__file__).stem}/inv_diffusion_{step:03d}.png', title=f'Step {step}', close=True)
```

| **Step 39** | **Step 38** | **Step 37** | **Step 36** | **Step 35** |
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| <img style="align-self:center;" src="swiss_roll/inv_diffusion_039.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_038.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_037.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_036.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_035.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> |
| **Step 34** | **Step 33** | **Step 32** | **Step 31** | **Step 30** |
| <img style="align-self:center;" src="swiss_roll/inv_diffusion_034.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_033.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_032.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_031.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_030.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> |
| **Step 29** | **Step 28** | **Step 27** | **Step 26** | **Step 25** |
| <img style="align-self:center;" src="swiss_roll/inv_diffusion_029.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_028.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_027.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_026.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_025.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> |
| **Step 24** | **Step 23** | **Step 22** | **Step 21** | **Step 20** |
| <img style="align-self:center;" src="swiss_roll/inv_diffusion_024.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_023.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_022.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_021.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_020.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> |
| **Step 19** | **Step 18** | **Step 17** | **Step 16** | **Step 15** |
| <img style="align-self:center;" src="swiss_roll/inv_diffusion_019.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_018.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_017.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_016.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_015.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> |
| **Step 14** | **Step 13** | **Step 12** | **Step 11** | **Step 10** |
| <img style="align-self:center;" src="swiss_roll/inv_diffusion_014.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_013.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_012.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_011.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_010.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> |
| **Step 9** | **Step 8** | **Step 7** | **Step 6** | **Step 5** |
| <img style="align-self:center;" src="swiss_roll/inv_diffusion_009.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_008.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_007.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_006.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_005.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> |
| **Step 4** | **Step 3** | **Step 2** | **Step 1** | **Step 0** |
| <img style="align-self:center;" src="swiss_roll/inv_diffusion_004.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_003.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_002.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_001.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> | <img style="align-self:center;" src="swiss_roll/inv_diffusion_000.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" close="True"/> |
