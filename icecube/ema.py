# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/callback.ema.ipynb.

# %% ../../nbs/callback.ema.ipynb 1
# EMA Callbacks are inspired by timm's ModelEmaV2: https://github.com/rwightman/pytorch-image-models/blob/main/timm/utils/model_ema.py
# PyTorch Image Models - Apache License 2.0 - Copyright (c) 2020 Ross Wightman

# %% ../../nbs/callback.ema.ipynb 3
from __future__ import annotations

from copy import deepcopy

from fastai.callback.core import Callback
from fastai.callback.fp16 import MixedPrecision
from fastai.callback.schedule import SchedCos, _Annealer

#from ..imports import *

import torch
from fastcore.basics import store_attr, noop, Self, hasattrs, partialler

# %% auto 0
__all__ = ['EMACallback', 'EMAWarmupCallback']

# %% ../../nbs/callback.ema.ipynb 6
class EMACallback(Callback):
    "Exponential Moving Average (EMA) of model weights with a fused update step"
    order,run_valid = MixedPrecision.order+1,False
    def __init__(self,
        decay:float=0.9998, # EMA decay value
        start= 0 , # Start EMA in percent of training steps (float) or epochs (int, index 0)
        ema_device:torch.device|str|None=None, # Device to store EMA weights. Defaults to model device
        validate_ema:bool=True, # Run validation metrics using EMA weights instead of model weights. If true, `ema_device` must match model device
        replace_weights:bool=False, # Replace model weights with EMA weights when finished training. If false, sets `Learner.model_ema` to EMA weights
        foreach:bool|None=None, # Fuse EMA update step with PyTorch ForEach methods or use a standard for loop. Defaults to true if PyTorch 1.12+ and Cuda device detected
        resume:bool=False, # Resume from EMA weights from previous training saved to `Learner.model_ema`
        all_parameters:bool=False, # Apply EMA step to all parameters or only those with `requires_grad`
        all_buffers:bool=False, # Apply EMA step to persistent model buffers or all buffers
        skip_ema:bool=True, # Skip EMA step if callbacks, such as GradientAccumulation or MixedPrecision, skip the Optimizer update step
    ):
        store_attr()
        self.inverse_decay = 1-decay
        #if self.foreach is None and ema_device is None:
        #    self.foreach = ismin_torch('1.12') and torch.cuda.is_available()

        #if self.foreach:
        #    if notmax_torch('1.12'):
        #        warn(f'EMACallback with foreach=True is untested on PyTorch {torch.__verson__}, recommended to use 1.12 or newer')

        if resume and self.start > 0:
            warn(f'Resuming from prior EMA weights but delaying EMA until {start=}')

    @torch.no_grad()
    def before_fit(self):
        if hasattr(self.learn, 'lr_finder') or hasattr(self.learn, "gather_preds"):
            self.run = False
            return

        self._do_ema, self._restore_ema = False, False

        if self.start >= 1 and isinstance(self.start, int):
            self.start = self.start/self.n_epoch
        if self.start >= 1:
            warn(f'EMA start {self.start} is equal or greater than one and will not start in this training run')

        if self.resume:
            self.ema_model = self.learn.model_ema.eval()
        else:
            self.ema_model = deepcopy(self.learn.model).eval()

        model_device = next(self.learn.model.parameters()).device
        self.ema_model.to(self.ema_device if self.ema_device is not None else model_device)
        ema_device = next(self.ema_model.parameters()).device

        self.model_tensors, self.ema_tensors = [], []
        for mt, et in zip(self.learn.model.parameters(), self.ema_model.parameters()):
            if self.all_parameters or mt.requires_grad:
                self.model_tensors.append(mt)
                self.ema_tensors.append(et)

        self.model_buffers, self.ema_buffers = [], []
        state_names = self.model.state_dict().keys()
        for (n, mb), (_, eb) in zip(self.learn.model.named_buffers(), self.ema_model.named_buffers()):
            if self.all_buffers or n in state_names:
                # foreach methods cannot convert non-floats back to original type and error out
                if self.foreach and torch.is_floating_point(mb):
                    self.model_tensors.append(mb)
                    self.ema_tensors.append(eb)
                else:
                    self.model_buffers.append(mb)
                    self.ema_buffers.append(mb)

        self._validate_ema = model_device == ema_device if self.validate_ema else False
        if self.foreach:
            assert model_device == ema_device, f"{ema_device=} must equal {model_device=} if using foreach"

    @torch.no_grad()
    def before_batch(self):
        if self.pct_train >= self.start:
            if self.start > 0 and not self._do_ema and not self.resume:
                self.ema_model.load_state_dict(self.learn.model.state_dict())
            self._do_ema = True

    def after_cancel_batch(self):
        # if a callback (such as GradientAccumulation) raises a CancelBatchException, don't do EMA step and potentially turn EMA back on
        if self.skip_ema:
            self._restore_ema = self._do_ema
            self._do_ema = False

    def after_cancel_step(self):
        # if a callback (such as MixedPrecision) raises a CancelStepException, don't do EMA step and potentially turn EMA back on
        if self.skip_ema:
            self._restore_ema = self._do_ema
            self._do_ema = False

    @torch.no_grad()
    def after_batch(self):
        if self._do_ema:
            if self.foreach:
                torch._foreach_mul_(self.ema_tensors, scalar=self.decay)
                torch._foreach_add_(self.ema_tensors, self.model_tensors, alpha=self.inverse_decay)
                # foreach methods cannot convert non-floats back to original type and error out
                for mb, eb in zip(self.model_buffers, self.ema_buffers):
                    eb.copy_(self.decay * eb + self.inverse_decay * mb)
            else:
                for mt, et in zip(self.model_tensors, self.ema_tensors):
                    et.copy_(self.decay * et + self.inverse_decay * mt)
        # handle a Cancel Exception while self._do_ema was set to True
        if self._restore_ema:
            self._do_ema = True
            self._restore_ema = False

    @torch.no_grad()
    def before_validate(self):
        if self._do_ema and self._validate_ema:
            self.temp_model = self.learn.model
            self.learn.model = self.ema_model

    @torch.no_grad()
    def after_validate(self):
        if self._do_ema and self._validate_ema:
            self.learn.model = self.temp_model

    def after_fit(self):
        if self.replace_weights:
            self.learn.model = self.ema_model
            self.ema_model = None
        else:
            self.learn.model_ema = self.ema_model

# %% ../../nbs/callback.ema.ipynb 9
class EMAWarmupCallback(EMACallback):
    "Exponential Moving Average (EMA) of model weights with a warmup schedule and fused update step"
    order,run_valid = MixedPrecision.order+1,False
    def __init__(self,
        start_decay:float=0.9, # Initial EMA decay value
        final_decay:float=0.9998, # Final EMA decay value
        start:Numeric=0, # Start EMA warmup in percent of training steps (float) or epochs (int, index 0)
        finish:Numeric=0.3, # Finish EMA warmup in percent of training steps (float) or epochs (int, index 0)
        schedule:Callable[..., _Annealer]=SchedCos, # EMA decay warmup schedule
        ema_device:torch.device|str|None=None, # Device to store EMA weights. Defaults to model device
        validate_ema:bool=True, # Run validation metrics using EMA weights instead of model weights. If true, `ema_device` must match model device
        replace_weights:bool=False, # Replace model weights with EMA weights when finished training. If false, set `Learner.model_ema` to EMA weights
        foreach:bool|None=None, # Fuse EMA update step with PyTorch ForEach methods or use a standard for loop. Defaults to true if PyTorch 1.12+ and Cuda device detected
        resume:bool=False, # Resume from EMA weights from previous training saved to `Learner.model_ema`
        all_parameters:bool=False, # Apply EMA step to all parameters or only those with `requires_grad`
        all_buffers:bool=False, # Apply EMA step to persistent model buffers or all buffers
        skip_ema:bool=True, # Skip EMA step if callbacks, such as GradientAccumulation or MixedPrecision, skip the Optimizer update step
        logger_callback:str='wandb', # Log EMA decay to `logger_callback` using `Callback.name` if available
    ):
        super().__init__(decay=final_decay, start=start, ema_device=ema_device,
                         validate_ema=validate_ema, replace_weights=replace_weights,
                         foreach=foreach, resume=resume, all_parameters=all_parameters,
                         all_buffers=all_buffers, skip_ema=skip_ema)
        store_attr(names='start_decay,final_decay,finish,logger_callback')
        self.schedule = schedule(start_decay, final_decay)

    def before_fit(self):
        if self.finish - self.start <= 0:
            warn(f'EMA Warmup start={self.start} is less or equal to final={self.epoch} which negates warmup')

        super().before_fit()

        if self.finish >= 1 and isinstance(self.finish, int):
            self.finish = self.finish/self.n_epoch
        if self.finish >= 1:
            warn(f'EMA Warmup finish {self.finish} is equal or greater than one and will not finish in this training run')

        if self.resume and self.n_epoch < self.finish*self.n_epoch:
            warn("Resuming EMA Warmup before the warmup is finished is not supported")

        # negate decay so at least one ema scheduling step will occur
        self.decay = -1*self.decay
        self.warmup_pct = 0.
        self._warmup_sched = 1/(len(self.dls.train) * self.n_epoch * (self.finish - self.start))

        self._log_ema_decay = getattr(self, f'_{self.logger_callback}_log_ema_decay', noop)
        self.has_logger = hasattr(self.learn, self.logger_callback) and self._log_ema_decay != noop

    def after_batch(self):
        if self._do_ema:
            if self.pct_train >= self.start and self.decay != self.final_decay:
                if self.pct_train >= self.finish:
                    self.decay = self.final_decay
                else:
                    self.decay = self.schedule(self.warmup_pct)
                    self.warmup_pct += self._warmup_sched
                self.inverse_decay = 1-self.decay

            super().after_batch()

        if self.has_logger:
            if self._do_ema:
                self._log_ema_decay(self.decay)
            else:
                self._log_ema_decay(0.)

# %% ../../nbs/callback.ema.ipynb 13
try:
    import wandb

    @patch
    def _wandb_log_ema_decay(self:EMAWarmupCallback, decay:float):
        wandb.log({'ema_decay': decay}, self.learn.wandb._wandb_step+1)
except:
    pass
