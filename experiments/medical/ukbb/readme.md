# UKBB experiments

## VI Experiments (`SVIExperiment`)

Command line args:
`num_svi_particles`: 
`use_cf_guide`:
`latent_dim`
`hidden_dim`

Variance predictions (`decoder_type`):
{`fixed_var`, `learned_var`, `independent_gaussian`, `sharedvar_multivariate_gaussian`, `multivariate_gaussian`}

```
python experiments/medical/trainer.py -e SVIExperiment -m ConditionalVISEM --default_root_dir /vol/biomedic2/np716/logdir/gemini/ukbb/ --downsample 3 --decoder_type fixed_var --train_batch_size 256 --gpus 0

python experiments/medical/trainer.py -e SVIExperiment -m ConditionalVISEM --default_root_dir /vol/biomedic2/np716/logdir/gemini/ukbb/ --downsample 3 --decoder_type fixed_var --train_batch_size 256 --gpus 0 --max_epochs 100

python experiments/medical/trainer.py -e SVIExperiment -m ConditionalVISEM --default_root_dir /vol/biomedic2/np716/logdir/gemini/ukbb/ --downsample 3 --decoder_type independent_gaussian --train_batch_size 256 --gpus 0

python experiments/medical/trainer.py -e SVIExperiment -m ConditionalVISEM --default_root_dir /vol/biomedic2/np716/logdir/gemini/ukbb/ --decoder_type fixed_var --gpus 0
python experiments/medical/trainer.py -e SVIExperiment -m ConditionalVISEM --default_root_dir /vol/biomedic2/np716/logdir/gemini/ukbb/ --decoder_type independent_gaussian --gpus 0

python experiments/medical/trainer.py -e SVIExperiment -m ConditionalVISEM --default_root_dir /vol/biomedic2/np716/logdir/gemini/ukbb/ --decoder_type fixed_var --latent_dim 256 --gpus 0
python experiments/medical/trainer.py -e SVIExperiment -m ConditionalVISEM --default_root_dir /vol/biomedic2/np716/logdir/gemini/ukbb/ --decoder_type independent_gaussian --latent_dim 256 --gpus 0
```