# Graph Network Simulator (GNS)
In this project, we use machine learning framework Graph Neural Network-based Simulators (GNS) as a surrogate model to predict granular and fluid flow dynamics.

## Run GNS

> Training GNS on simulation data
```shell
python3 -m gns.train --data_path="<input-training-data-path>" --model_path="<path-to-load-save-model-file>" --output_path="<path-to-save-output>" -ntraining_steps=100
```

> Resume training

To resume training specify `model_file` and `train_state_file`:

```shell
python3 -m gns.train --data_path="<input-training-data-path>" --model_path="<path-to-load-save-model-file>" --output_path="<path-to-save-output>"  --model_file="model.pt" --train_state_file="train_state.pt" -ntraining_steps=100
```

> Rollout prediction
```shell
python3 -m gns.train --mode="rollout" --data_path="<input-data-path>" --model_path="<path-to-load-save-model-file>" --output_path="<path-to-save-output>" --model_file="model.pt" --train_state_file="train_state.pt"
```

> Render
```shell
python3 -m gns.render_rollout --output_mode="gif" --rollout_dir="<path-containing-rollout-file>" --rollout_name="<name-of-rollout-file>"
```
