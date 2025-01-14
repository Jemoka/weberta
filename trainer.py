# common standard library utilities
import os
import sys
import time
import json
import math
import random
from random import Random

from pathlib import Path
from argparse import Namespace

# machine learning and data utilities
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR

# huggingface
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

# MLOps
import wandb
from accelerate import Accelerator

# logging
from loguru import logger

# our stuff
from model import *
from data import *

R = Random(7)

class Trainer:
    def __init__(self, args):
        # set up the trainer
        self.args = args
        self.accelerator = Accelerator(log_with="wandb")
        self.accelerator.init_trackers(
            project_name="adventure", 
            config=vars(args),
            init_kwargs={"wandb": {"mode": None if args.wandb else "disabled",
                                   "name": args.experiment}},
        )

        # ...and the output path
        save_dir = Path(args.out_dir) / args.experiment
        save_dir.mkdir(parents=True, exist_ok=True)

        self.save_dir = str(save_dir / "checkpoint")
        self.best_dir = str(save_dir / "best")

        # set up models
        self.model_config = AutoConfig.from_pretrained(args.base)
        self.model = AutoModelForMaskedLM.from_config(self.model_config)

        # set up data
        (self.train_dl, self.val_dl) = load_dls(args)

        # leave blank
        # this will exist if we are resuming from checkpoint
        self.train_dl_skipped = None 

        # optimizer
        self.optim = AdamW(self.model.parameters(), lr=args.lr)

        # compute training size + the counter (useful for mid-checkpoint recovery) 
        self.total_batches = int(1170381//args.batch_size) # TODO hard coding david's dataset size
        self.global_step_counter_ = 0
        self.best_val_score_ = float("-inf") # "score" means higher is better 

        # buliding scheduler
        warmup_steps = int(args.warmup_pct*self.total_batches)
        scheduler1 = LinearLR(self.optim, start_factor=1e-20, end_factor=1, total_iters=warmup_steps)
        scheduler2 = LinearLR(self.optim, start_factor=1, end_factor=0, total_iters=(self.total_batches-warmup_steps))
        self.scheduler = SequentialLR(self.optim, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])

        # weeeeeeeeeeee
        (self.model, self.optim, self.train_dl, self.val_dl, self.scheduler) = self.accelerator.prepare(
            self.model, self.optim, self.train_dl, self.val_dl, self.scheduler)
        if self.accelerator.is_main_process:
            wandb.watch(self.model)

    def train(self):
        for eid in range(self.args.epochs):
            if self.global_step_counter_ >= ((eid+1)*self.total_batches):
                logger.debug("SKIPPING EPOCH {} due to global step count...", eid)
                continue

            self.epoch()

        self.finish()

    def finish(self):
        self.accelerator.end_training()

    def val(self):
        all_losses = []
        with torch.inference_mode():
            for batch in self.val_dl:
                with torch.inference_mode():
                    loss = self.model(**batch).loss
                    all_losses.append(self.gather(loss))

        metrics = {"val/loss": sum(all_losses)/len(all_losses)}
        score = 1/(sum(all_losses)/len(all_losses))

        return score.cpu().item(), metrics

    def epoch(self):
        if self.accelerator.is_main_process:
            logger.info("BEGIN EPOCH")

        # because sometimes the load function may skip some epochs
        dl = self.train_dl if not self.train_dl_skipped else self.train_dl_skipped
        for indx, i in enumerate(dl):

            # take a step
            loss, train_metrics = self.step(i)
            train_metrics["train/lr"] = self.optim.param_groups[0]["lr"]

            # perform logging, and then increment
            # (we do this because global_step_counter_
            #  is useful not as the # of steps but how
            #  many we need to skip for warm start)
            if indx % self.args.report_interval == 0 and indx != 0:
                self.accelerator.log(train_metrics, step=self.global_step_counter_)
                if self.accelerator.is_main_process:
                    logger.info("TRAIN | {}/{} | loss {}", self.global_step_counter_,
                                self.total_batches*self.args.epochs, loss)
            self.global_step_counter_ += 1

            logger.debug("STEP | {} | {}", indx, train_metrics)

            # save a checkpoint, if needed
            if (indx % self.args.checkpoint_interval == 0 and indx != 0 and
                self.accelerator.is_main_process):
                self.save(self.save_dir)
            # perform validation and save a checkpoint, if needed
            if indx % self.args.validation_interval == 0 and indx != 0:
                score, val_metrics = self.val()
                self.accelerator.log(val_metrics, step=self.global_step_counter_)
                if self.accelerator.is_main_process:
                    logger.info("VAL | {} | score {}", self.global_step_counter_, score)

                if score > self.best_val_score_ and self.accelerator.is_main_process:
                    logger.info("VAL | BEST SCORE | score {}", score)
                    self.best_val_score_ = score
                    self.save(self.best_dir)

        # we are done using the skipped DL since we finished the remaining batch
        self.train_dl_skipped = None

    def step(self, batch):
        loss = self.model(**batch).loss

        self.accelerator.backward(loss)
        self.optim.step()
        self.scheduler.step()
        self.optim.zero_grad()

        loss = self.gather(loss).cpu().item() 
        metrics = { "train/loss": loss }

        return loss, metrics
        

    def load(self, path):
        self.accelerator.load_state(path)
        with open(os.path.join(path, "config.json"), 'r') as df:
            data = json.load(df)

        self.args = Namespace(**data.get("config", {}))
        self.global_step_counter_ = data.get("steps", 0)
        self.best_val_score_ = data.get("score", 0)

        # skip batches
        self.train_dl_skipped = self.accelerator.skip_first_batches(self.train_dl,
                                                                    self.global_step_counter_ % self.total_batches)

    def save(self, path):
        logger.debug("CHECKPOINT | saving checkpoint at {}", path)
        self.accelerator.save_state(path, safe_serialization=False)
        with open(os.path.join(path, "config.json"), 'w') as df:
            json.dump({
                "config": vars(self.args),
                "steps": self.global_step_counter_,
                "score": self.best_val_score_
            }, df)

    @classmethod
    def from_pretrained(cls, path, disable_wandb=True):
        with open(os.path.join(path, "config.json"), 'r') as df:
            data = json.load(df)
        args = Namespace(**data.get("config", {}))
        args.wandb = False
        new = cls(args)
        new.load(path)

        if disable_wandb:
            new.args.wandb = False

        return new

    @property
    def device(self):
        return self.accelerator.device

    def gather(self, n):
        result = self.accelerator.gather(n)
        if isinstance(result, list):
            return sum(result)/len(result)
        else:
            return result.mean()
    

