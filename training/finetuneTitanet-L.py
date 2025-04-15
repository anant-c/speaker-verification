import os
import glob
import subprocess
import tarfile
import wget
import nemo
# NeMo's ASR collection - This collection contains complete ASR models and
# building blocks (modules) for ASR
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
import torch
import lightning.pytorch as pl
from nemo.utils.exp_manager import exp_manager
import torch


MODEL_CONFIG = os.path.join('conf/titanet-finetune.yaml')
finetune_config = OmegaConf.load(MODEL_CONFIG)
print(OmegaConf.to_yaml(finetune_config))

train_manifest = os.path.join('datasetFinal/DEV/chunk/train.json')
finetune_config.model.train_ds.manifest_filepath = train_manifest
finetune_config.model.validation_ds.manifest_filepath = train_manifest
finetune_config.model.decoder.num_classes = 50


accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'


trainer_config = OmegaConf.create(dict(
    devices=1,
    accelerator=accelerator,
    max_epochs=50,
    max_steps=-1,  # computed at runtime if not set
    num_nodes=1,
    accumulate_grad_batches=1,
    enable_checkpointing=False,  # Provided by exp_manager
    logger=False,  # Provided by exp_manager
    log_every_n_steps=1,  # Interval of logging.
    val_check_interval=1.0,  # Set to 0.25 to check 4 times per epoch, or an int for number of iterations
))
print(OmegaConf.to_yaml(trainer_config))


trainer_finetune = pl.Trainer(**trainer_config)

log_dir_finetune = exp_manager(trainer_finetune, finetune_config.get("exp_manager", None))
print(log_dir_finetune)

speaker_model = nemo_asr.models.EncDecSpeakerLabelModel(cfg=finetune_config.model, trainer=trainer_finetune)
speaker_model.maybe_init_from_pretrained_checkpoint(finetune_config)

trainer_finetune.fit(speaker_model)

# saving the model
restored_model.save_to(os.path.join(log_dir_finetune, '..',"titanet-large-finetuneIMSV.nemo"))
