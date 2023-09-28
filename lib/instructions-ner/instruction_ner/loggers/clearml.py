from argparse import Namespace
from pathlib import Path
from random import randint
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import clearml
from _weakref import ReferenceType
from clearml import OutputModel, Task
from faker import Faker
from pytorch_lightning.callbacks import Checkpoint
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities.logger import (
    _add_prefix,
    _convert_params,
    _flatten_dict,
    _sanitize_callable_params,
)


class ClearMLLogger(Logger):
    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: str = ".",
        id: Optional[str] = None,
        project: str = "lightning_logs",
        log_model: Union[str, bool] = False,
        prefix: str = "",
        output_uri: str = "",
        agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
        agg_default_func: Optional[Callable[[Sequence[float]], float]] = None,
    ) -> None:
        if clearml is None:
            raise ModuleNotFoundError(
                "You want to use `clearml` logger which is not installed yet,"
                " install it with `pip install clearml`."  # pragma: no-cover
            )
        super().__init__(agg_key_funcs=agg_key_funcs, agg_default_func=agg_default_func)
        self._log_model = log_model
        self._prefix = prefix
        self._logged_model_time: Dict[str, float] = {}
        # set clearml init arguments
        if name is None:
            fake = Faker()
            name = "-".join(fake.bs().split(" ")[:2]) + "-" + str(randint(10, 999))  # nosec
        self._clearml_init: Dict[str, Any] = dict(
            task_name=name, project_name=project, auto_connect_frameworks={"pytorch": False}
        )
        self._checkpoint_callback: Optional["ReferenceType[Checkpoint]"] = None
        self._experiment = Task.init(**self._clearml_init, output_uri=output_uri)
        self.output_model = OutputModel(task=self._experiment)
        # extract parameters
        self._project = self._clearml_init.get("project_name")
        self._save_dir = save_dir
        self._name = self._clearml_init.get("task_name")
        self._id = id

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        # args needed to reload correct experiment
        if self._experiment is not None:
            state["_id"] = getattr(self._experiment, "id", None)
            state["_attach_id"] = getattr(self._experiment, "_attach_id", None)
            state["_name"] = self._experiment.name
        # cannot be pickled
        state["_experiment"] = None
        return state

    @property  # type: ignore[misc]
    @rank_zero_experiment
    def experiment(self):
        r"""

        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.module.LightningModule` do the following.

        Example::

        .. code-block:: python

            self.logger.experiment.some_wandb_function()

        """
        self._experiment = Task.init(**self._clearml_init)
        return self._experiment

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        if step is not None:
            for scalar_name, scalar_value in metrics.items():
                if not isinstance(scalar_value, dict):
                    self._experiment.get_logger().report_scalar(
                        title=scalar_name, series=scalar_name, value=scalar_value, iteration=step
                    )

    def log_hyperparams(
        self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any
    ) -> None:
        params = _convert_params(params)
        params = _flatten_dict(params)
        params = _sanitize_callable_params(params)
        self._experiment.connect(params)

    @property
    def save_dir(self) -> Optional[str]:
        """Gets the save directory.

        Returns:
            The path to the save directory.
        """
        return self._save_dir

    @property
    def name(self) -> Optional[str]:
        """The project name of this experiment.

        Returns:
            The name of the project the current experiment belongs to.
        """
        return self._project

    @property
    def version(self) -> Optional[str]:
        """Gets the id of the experiment.

        Returns:
            The id of the experiment if the experiment exists else the id given to the constructor.
        """
        # don't create an experiment if we don't have one
        return self._experiment.id if self._experiment else self._id

    def after_save_checkpoint(self, checkpoint_callback: ReferenceType[Checkpoint]) -> None:
        # log checkpoints as artifacts
        if (
            self._log_model == "all"
            or self._log_model is True
            and hasattr(checkpoint_callback, "save_top_k")
            and checkpoint_callback.save_top_k == -1  # type: ignore
        ):
            self._scan_and_log_checkpoints(checkpoint_callback)
        elif self._log_model is True:
            self._checkpoint_callback = checkpoint_callback

    def finalize(self, status: str) -> None:
        # log checkpoints as artifacts
        if self._checkpoint_callback:
            self._scan_and_log_checkpoints(self._checkpoint_callback)

    def _scan_and_log_checkpoints(self, checkpoint_callback: "ReferenceType[Checkpoint]") -> None:
        # get checkpoints to be saved with associated score
        checkpoints = dict()
        if hasattr(checkpoint_callback, "last_model_path") and hasattr(
            checkpoint_callback, "current_score"
        ):
            checkpoints[checkpoint_callback.last_model_path] = (  # type: ignore
                checkpoint_callback.current_score,  # type: ignore
                "latest",
            )

        if hasattr(checkpoint_callback, "best_model_path") and hasattr(
            checkpoint_callback, "best_model_score"
        ):
            checkpoints[checkpoint_callback.best_model_path] = (  # type: ignore
                checkpoint_callback.best_model_score,  # type: ignore
                "best",
            )

        if hasattr(checkpoint_callback, "best_k_models"):
            for key, value in checkpoint_callback.best_k_models.items():  # type: ignore
                checkpoints[key] = (value, "best_k")

        checkpoints_list = sorted(
            (Path(p).stat().st_mtime, p, s, tag)
            for p, (s, tag) in checkpoints.items()
            if Path(p).is_file()
        )
        checkpoints_list = [
            c
            for c in checkpoints_list
            if c[1] not in self._logged_model_time.keys() or self._logged_model_time[c[1]] < c[0]
        ]

        # log iteratively all new checkpoints
        for t, p, s, tag in checkpoints_list:
            self.output_model.update_weights(weights_filename=p)
            # remember logged models - timestamp needed in case filename didn't change (lastkckpt
            # or custom name)
            self._logged_model_time[p] = t
