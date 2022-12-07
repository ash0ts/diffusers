# Based on https://github.com/huggingface/accelerate/blob/d2e804f69d904db237d8ebdbcdc0d00fb52731ab/src/accelerate/tracking.py
from accelerate.utils.imports import is_wandb_available

if is_wandb_available():
    import wandb


class GeneralExtendedTracker:
    """
    An extended Tracker class to be used for supplemental logging integration implementations.
    """

    @abstractmethod
    def log_images(self, epoch: int, global_step: int, images_processed: list):
        """
        Should log a list of images to provided tracking API
        """
        pass

    @abstractmethod
    def log_model(self, epoch: int, global_step: int, model_dir):
        """
        Should log the contents of a directory containing saved model assets to provided tracking API
        """
        pass

    @abstractproperty
    def finalize(self):
        """
        Should flush any internal dataclasses such as tables to provided tracking API
        """
        pass


class ExtendedWandBTracker(GeneralExtendedTracker):
    """
    A `Tracker` class that extends support for `wandb`. Should inherit the tracker from accelerate
    Args:
        run :
            The run already initialized by accelerate retrieved typically by `accelerator.get_tracker('wandb')`
        kwargs:
            Additional key word arguments.
    """

    wandb_table = wandb.Table(columns=["epoch", "global_step", "generated_images"])

    def __init__(self, run, **kwargs):
        self.run = run

    def log_images(self, epoch: int, global_step: int, images_processed: list):
        wandb_images = [wandb.Image(i) for i in images_processed]
        self.wandb_table.add_data(epoch, global_step, wandb_images)

        self.run.log(
            {
                "generated_images": wandb_images,
            },
            step=global_step,
        )

    def log_model(self, epoch: int, global_step: int, model_dir):
        model_artifact = wandb.Artifact(f"{self.run.id}-{model_dir}", type="model")
        model_artifact.add_dir(model_dir)
        self.run.log_artifact(model_artifact, aliases=[f"step_{global_step}", f"epoch_{epoch}"])

    def finalize(self):
        self.run.log({"generated_samples_table": self.wandb_table})
