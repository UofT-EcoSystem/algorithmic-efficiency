# Getting Started with MNIST

There are two parts of this guide: (1) Install and test MNIST Docker container and (2) getting familiar with the code base.

## 1) Install and test MNIST Docker container
 
Assuming you have a Ubuntu machine with docker. A GPU is optional. I use VSCode Remote to connect to a server for development.

1. Checkout the code

```bash
git clone https://github.com/UofT-EcoSystem/algorithmic-efficiency
cd algorithmic-efficiency
```

2.  Build the docker container

```bash
sudo docker build -t algorithmic-efficiency:latest \
    --build-arg UID=$UID `# share the same user ID inside and outside container to make file permissions across volume mounts seamless` \
    --build-arg ACCELERATOR=gpu `# select ACCELERATOR to be "gpu" or "cpu"` \
    .
```

3. Run the container

```bash
sudo docker run -it \
    --gpus all --ipc=host `# connect the GPUs to the container (remove if using CPU)` \
    -v $PWD:/home/ubuntu/algorithmic-efficiency `# volume mount the code (located at $PWD) into the container so it is synced` \
    --name algorithmic-efficiency \
    algorithmic-efficiency
```

- (For future reference) If the docker container stops and you want to run it again without creating a new container

```bash
sudo docker start algorithmic-efficiency
sudo docker exec -it algorithmic-efficiency /bin/bash
```

4.  Run Jax MNIST MLP

```bash
python3 algorithmic_efficiency/submission_runner.py --framework=jax --workload=mnist_jax --submission_path=baselines/mnist/mnist_jax/submission.py --tuning_search_space=baselines/mnist/tuning_search_space.json
```

5.  It is working if you see output like this
```python
I0218 16:15:49.943939 139713788225344 submission_runner.py:207] 2.73s   1       {'accuracy': 0.10870000720024109, 'loss': 2.659390449523926}
```

## 2) Getting Familiar With The Code Base

I've included some important code sections below.

1. The definition of the MLP model

```python
class _Model(nn.Module):
  @nn.compact
  def __call__(self, x: spec.Tensor, train: bool):
    del train
    input_size = 28 * 28
    num_hidden = 128
    num_classes = 10
    x = x.reshape((x.shape[0], input_size))  # Flatten.
    x = nn.Dense(features=num_hidden, use_bias=True)(x)
    x = nn.sigmoid(x)
    x = nn.Dense(features=num_classes, use_bias=True)(x)
    x = nn.log_softmax(x)
    return x
```
From: https://github.com/UofT-EcoSystem/algorithmic-efficiency/blob/main/algorithmic_efficiency/workloads/mnist/mnist_jax/workload.py#L17

2. The data loader

```python
    ds = tfds.load('mnist', split=split)
    ds = ds.cache()
    ds = ds.map(lambda x: (self._normalize(x['image']), x['label']))
    if split == 'train':
      ds = ds.shuffle(1024, seed=data_rng[0])
      ds = ds.repeat()
    ds = ds.batch(batch_size)
    return tfds.as_numpy(ds)
```
From: https://github.com/UofT-EcoSystem/algorithmic-efficiency/blob/main/algorithmic_efficiency/workloads/mnist/mnist_jax/workload.py#L43

3. The Adam optimizer

```python
def optimizer(hyperparameters):
  opt_init_fn, opt_update_fn = optax.chain(
      optax.scale_by_adam(
          b1=1.0 - hyperparameters.one_minus_beta_1,
          b2=0.999,
          eps=hyperparameters.epsilon),
      optax.scale(-hyperparameters.learning_rate))
  return opt_init_fn, opt_update_fn
```
From: https://github.com/UofT-EcoSystem/algorithmic-efficiency/blob/main/baselines/mnist/mnist_jax/submission.py#L19

4. The optimizer's hyperparameters are chosen from a range of valid values. 

```json
{
    "learning_rate": {"min": 1e-4, "max": 1e-2, "scaling": "log"},
    "one_minus_beta_1": {"min": 0.9, "max": 0.999, "scaling": "log"},
    "epsilon": {"feasible_points": [1e-8, 1e-5, 1e-3]}
}
```
From: https://github.com/UofT-EcoSystem/algorithmic-efficiency/blob/main/baselines/mnist/tuning_search_space.json

The benchmark will average the scores of 20 runs with different hyperparameters chosen from the above range. Instead of 20 you can do just 1 run by adding `--num_tuning_trials=1` to the `python3 algorithmic_efficiency/submission_runner.py` command.

5. The main training loop is here. I have abbreviated the code with (...)
```python
  # Workload setup.
  logging.info('Initializing dataset.')
  input_queue = workload.build_input_queue(
      data_rng, 'train', data_dir=data_dir, batch_size=batch_size)
  logging.info('Initializing model.')
  model_params, model_state = workload.init_model_fn(model_init_rng)
  logging.info('Initializing optimizer.')
  optimizer_state = init_optimizer_state(workload, model_params, model_state,
                                         hyperparameters, opt_init_rng)
  (...)
  logging.info('Starting training loop.')
  while (is_time_remaining and not goal_reached and not training_complete):
    (...)
    optimizer_state, model_params, model_state = update_params(
      workload=workload,
      current_param_container=model_params,
      current_params_types=workload.model_params_types(),
      model_state=model_state,
      hyperparameters=hyperparameters,
      input_batch=selected_train_input_batch,
      label_batch=selected_train_label_batch,
      loss_type=workload.loss_type,
      optimizer_state=optimizer_state,
      eval_results=eval_results,
      global_step=global_step,
      rng=update_rng)
      (...)
      if (current_time - last_eval_time >= workload.eval_period_time_sec or
        training_complete):
        latest_eval_result = workload.eval_model(model_params, model_state,
                                               eval_rng, data_dir)
      logging.info(f'{current_time - global_start_time:.2f}s\t{global_step}'
                   f'\t{latest_eval_result}')
```
From: https://github.com/UofT-EcoSystem/algorithmic-efficiency/blob/main/algorithmic_efficiency/submission_runner.py#L153