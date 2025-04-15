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


train_manifest = os.path.join('datasetFinal/DEV/chunk/train.json')
validation_manifest = os.path.join('datasetFinal/DEV/chunk/dev.json')
test_manifest = os.path.join('datasetFinal/DEV/chunk/dev.json')

MODEL_CONFIG = os.path.join('conf/titanet-large.yaml')
config = OmegaConf.load(MODEL_CONFIG)
print(OmegaConf.to_yaml(config))

print(OmegaConf.to_yaml(config.model.train_ds))
print(OmegaConf.to_yaml(config.model.validation_ds))


config.model.train_ds.manifest_filepath = train_manifest
config.model.validation_ds.manifest_filepath = validation_manifest


config.model.decoder.num_classes = 50

print("Trainer config - \n")
print(OmegaConf.to_yaml(config.trainer))

# Let us modify some trainer configs for this demo
# Checks if we have GPU available and uses it
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
config.trainer.devices = 1
config.trainer.accelerator = accelerator

# Reduces maximum number of epochs to 5 for quick demonstration
config.trainer.max_epochs = 50

# Remove distributed training flags
config.trainer.strategy = 'auto'

# Remove augmentations
config.model.train_ds.augmentor=None

trainer = pl.Trainer(**config.trainer)

log_dir = exp_manager(trainer, config.get("exp_manager", None))
# The log_dir provides a path to the current logging directory for easy access
print(log_dir)

speaker_model = nemo_asr.models.EncDecSpeakerLabelModel(cfg=config.model, trainer=trainer)


torch.cuda.empty_cache()

trainer.fit(speaker_model)


checkpoint_dir = os.path.join(log_dir, 'checkpoints')
checkpoint_paths = list(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")))
print(checkpoint_paths)

final_checkpoint = list(filter(lambda x: "-last.ckpt" in x, checkpoint_paths))[0]
print(final_checkpoint)

#optional (my modification)
best_checkpoint = min(checkpoint_paths, key=lambda x: float(x.split("val_loss=")[1].split("-epoch=")[0]))
print(best_checkpoint)

restored_model = nemo_asr.models.EncDecSpeakerLabelModel.load_from_checkpoint(final_checkpoint)


restored_model.save_to(os.path.join("titanet-large-IMSV.nemo"))

# verification part


verification_model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from("titanet-large-IMSV.nemo")

print("verificationmodel train kia saved hoga titanet-large-IMSV.nemo name se ")
