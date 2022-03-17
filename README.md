# Models for BIB


The model is trained in two stages. The first stage is a pretraineing phase where an image encoder is trained using the [ATC algorithm](https://arxiv.org/abs/2009.08319).
In the second stage, the model is trained using behavior cloning.

## Setup
Setup the python environment and install the required libraries specified in the requirements file.

```
python -m virtualenv /path/to/env
source /path/to/env/bin/activate
pip install -r requirements.txt
```

## Training ATC
Train the ATC model using the following command.

```
python train_atc.py --data_path /path/to/train_data/ --batch_size 512 --cache 1 --size 84 --max_epochs 100 --val_check_interval 0.5 --gpus 1 --gradient_clip_val  10 --num_workers 16 --track_grad_norm 2 --num_sanity_val_steps 2 --random_shift 1 --stochastic_weight_avg True 
```

## Training Behavior Cloning

After training, the behavior cloning model is trained using the following command.

```
python train_tom.py --model_type bcmlp --data_path /path/to/train_data/ --seed 1 --batch_size 32 --max_epochs 1000 --gpus 1 --num_workers 12  --stochastic_weight_avg True --lr 1e-4 --check_val_every_n_epoch 1 --max_epochs 1000 --process_data 1 --track_grad_norm 2 --gradient_clip_val 10
```

## Evaluting the model
Once the model is trained, use the following command to evaluate the model.

```
python test_bc.py --data_path /path/to/train_data/ --ckpt /path/to/model/ --process_data 1 --model_type bcmlp
```

### Hardware setup
All models were trained on an NVIDIA 1080Ti GPU.

