# MorphoMNIST Experiments

## VI Experiments (`SVIExperiment`)

Command line args:
`num_svi_particles`: 
`use_cf_guide`:
`latent_dim`
`hidden_dim`

Models:
{`ConditionalVISEM`, `ConditionalDecoderVISEM`, `IndependentVISEM`}
(& {`ConditionalSTNDecoderVISEM`, `ConditionalSTNVISEM`})

Variance predictions (`decoder_type`):
{`fixed_var`, `learned_var`, `independent_gaussian`, `sharedvar_multivariate_gaussian`, `multivariate_gaussian`}


All digit classes:
`data_dir`: `/vol/biomedic2/np716/data/gemini/synthetic/thickness_intensity/all_scale05/`

```
python experiments/morphomnist/trainer.py -e SVIExperiment -m ConditionalVISEM --data_dir /vol/biomedic2/np716/data/gemini/synthetic/thickness_intensity/all_scale05/ --default_save_path /vol/biomedic2/np716/logdir/gemini/morphomnist/all/ --decoder_type fixed_var --gpus 0
python experiments/morphomnist/trainer.py -e SVIExperiment -m ConditionalVISEM --data_dir /vol/biomedic2/np716/data/gemini/synthetic/thickness_intensity/all_scale05/ --default_save_path /vol/biomedic2/np716/logdir/gemini/morphomnist/all/ --decoder_type independent_gaussian --gpus 0

python experiments/morphomnist/trainer.py -e SVIExperiment -m ConditionalDecoderVISEM --data_dir /vol/biomedic2/np716/data/gemini/synthetic/thickness_intensity/all_scale05/ --default_save_path /vol/biomedic2/np716/logdir/gemini/morphomnist/all/ --decoder_type fixed_var --gpus 0
python experiments/morphomnist/trainer.py -e SVIExperiment -m ConditionalDecoderVISEM --data_dir /vol/biomedic2/np716/data/gemini/synthetic/thickness_intensity/all_scale05/ --default_save_path /vol/biomedic2/np716/logdir/gemini/morphomnist/all/ --decoder_type independent_gaussian --gpus 0

python experiments/morphomnist/trainer.py -e SVIExperiment -m IndependentVISEM --data_dir /vol/biomedic2/np716/data/gemini/synthetic/thickness_intensity/all_scale05/ --default_save_path /vol/biomedic2/np716/logdir/gemini/morphomnist/all/ --decoder_type fixed_var --gpus 0
python experiments/morphomnist/trainer.py -e SVIExperiment -m IndependentVISEM --data_dir /vol/biomedic2/np716/data/gemini/synthetic/thickness_intensity/all_scale05/ --default_save_path /vol/biomedic2/np716/logdir/gemini/morphomnist/all/ --decoder_type independent_gaussian --gpus 0

python experiments/morphomnist/trainer.py -e SVIExperiment -m ConditionalVISEM --data_dir /vol/biomedic2/np716/data/gemini/synthetic/thickness_intensity/all_scale05/ --default_save_path /vol/biomedic2/np716/logdir/gemini/morphomnist/all/ --decoder_type fixed_var --gpus 0 --use_cf_guide
python experiments/morphomnist/trainer.py -e SVIExperiment -m ConditionalVISEM --data_dir /vol/biomedic2/np716/data/gemini/synthetic/thickness_intensity/all_scale05/ --default_save_path /vol/biomedic2/np716/logdir/gemini/morphomnist/all/ --decoder_type independent_gaussian --gpus 0 --use_cf_guide
```

'2's:
`data_dir`: `/vol/biomedic2/np716/data/gemini/synthetic/thickness_intensity/2_scale05/`

```
python experiments/morphomnist/trainer.py -e SVIExperiment -m ConditionalVISEM --default_save_path /vol/biomedic2/np716/logdir/gemini/morphomnist/2/ --decoder_type fixed_var --gpus 0
python experiments/morphomnist/trainer.py -e SVIExperiment -m ConditionalVISEM --default_save_path /vol/biomedic2/np716/logdir/gemini/morphomnist/2/ --decoder_type independent_gaussian --gpus 0

python experiments/morphomnist/trainer.py -e SVIExperiment -m ConditionalDecoderVISEM --default_save_path /vol/biomedic2/np716/logdir/gemini/morphomnist/2/ --decoder_type fixed_var --gpus 0
python experiments/morphomnist/trainer.py -e SVIExperiment -m ConditionalDecoderVISEM --default_save_path /vol/biomedic2/np716/logdir/gemini/morphomnist/2/ --decoder_type independent_gaussian --gpus 0

python experiments/morphomnist/trainer.py -e SVIExperiment -m IndependentVISEM --default_save_path /vol/biomedic2/np716/logdir/gemini/morphomnist/2/ --decoder_type fixed_var --gpus 0
python experiments/morphomnist/trainer.py -e SVIExperiment -m IndependentVISEM --default_save_path /vol/biomedic2/np716/logdir/gemini/morphomnist/2/ --decoder_type independent_gaussian --gpus 0

python experiments/morphomnist/trainer.py -e SVIExperiment -m ConditionalVISEM --default_save_path /vol/biomedic2/np716/logdir/gemini/morphomnist/2/ --decoder_type fixed_var --use_cf_guide --gpus 0 
python experiments/morphomnist/trainer.py -e SVIExperiment -m ConditionalVISEM --default_save_path /vol/biomedic2/np716/logdir/gemini/morphomnist/2/ --decoder_type independent_gaussian --use_cf_guide --gpus 0
```

## NF Experiments