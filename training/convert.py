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

final_checkpoint = "modelsIMSV/titanet-large-finetuneIMSV.ckpt"

restored_model = nemo_asr.models.EncDecSpeakerLabelModel.load_from_checkpoint(final_checkpoint)


restored_model.save_to(os.path.join("titanet-large-finetuneIMSV.nemo"))
