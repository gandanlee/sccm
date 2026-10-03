import os
import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from loguru import logger
import gc

import sccm

class CheckPoint:
    def __init__(self, dir=None, name="tmp"):
        self.name = name
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)
        # Best-checkpoint tracking (lazily initialized from disk on first save_best).
        self.best_metric = None

    def save(
        self,
        model,
        optimizer,
        lr_scheduler,
        n,
        ):
        if sccm.RANK == 0:
            assert model is not None
            if isinstance(model, (DataParallel, DistributedDataParallel)):
                model = model.module
            states = {
                "model": model.state_dict(),
                "n": n,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }
            torch.save(states, self.dir + self.name + f"_latest.pth")
            logger.info(f"Saved states {list(states.keys())}, at step {n}")

    def save_best(
        self,
        model,
        optimizer,
        lr_scheduler,
        n,
        metric,
        metric_name="pck_0p35",
        ):
        """Save <name>_best.pth iff `metric` beats the best seen so far.

        The best value survives restarts: on first call we read it back from any
        existing _best.pth so a resumed run does not overwrite a better checkpoint.
        """
        if sccm.RANK != 0 or metric is None:
            return
        best_path = self.dir + self.name + f"_best.pth"
        if self.best_metric is None:
            self.best_metric = float("-inf")
            if os.path.exists(best_path):
                try:
                    prev = torch.load(best_path, map_location="cpu")
                    self.best_metric = float(prev.get("best_metric", float("-inf")))
                except Exception as e:
                    print(f"Failed to read best metric from {best_path}: {e}")
        if metric <= self.best_metric:
            return
        self.best_metric = metric
        m = model
        if isinstance(m, (DataParallel, DistributedDataParallel)):
            m = m.module
        states = {
            "model": m.state_dict(),
            "n": n,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "best_metric": metric,
            "best_metric_name": metric_name,
        }
        torch.save(states, best_path)
        logger.info(f"Saved BEST checkpoint ({metric_name}={metric:.4f}) at step {n}")

    def load(
        self,
        model,
        optimizer,
        lr_scheduler,
        n,
        ):
        if os.path.exists(self.dir + self.name + f"_latest.pth"):
            states = torch.load(self.dir + self.name + f"_latest.pth")
            if "model" in states:
                model.load_state_dict(states["model"])
            if "n" in states:
                n = states["n"] if states["n"] else n
            if "optimizer" in states:
                try:
                    optimizer.load_state_dict(states["optimizer"])
                except Exception as e:
                    print(f"Failed to load states for optimizer, with error {e}")
            if "lr_scheduler" in states:
                lr_scheduler.load_state_dict(states["lr_scheduler"])
            if sccm.RANK == 0:
                print(f"Loaded states {list(states.keys())}, at step {n}")
            del states
            gc.collect()
            torch.cuda.empty_cache()
        return model, optimizer, lr_scheduler, n