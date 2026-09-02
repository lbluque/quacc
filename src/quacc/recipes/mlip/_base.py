"""Common utility functions for universal machine-learned interatomic potentials."""

from __future__ import annotations

from functools import lru_cache, wraps
from importlib.util import find_spec
from logging import getLogger
from typing import TYPE_CHECKING

from monty.dev import requires

has_frozen = bool(find_spec("frozendict"))
has_torch = bool(find_spec("torch"))

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal

    from ase.calculators.calculator import BaseCalculator


LOGGER = getLogger(__name__)


@requires(has_frozen, "frozendict must be installed. Run pip install frozendict.")
def freezeargs(func: Callable) -> Callable:
    """
    Convert a mutable dictionary into immutable.
    Useful to make sure dictionary args are compatible with cache
    From https://stackoverflow.com/a/53394430

    Parameters
    ----------
    func
        Function to be wrapped.

    Returns
    -------
    Callable
        Wrapped function with frozen dictionary arguments.
    """
    from frozendict import frozendict

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        args = (frozendict(arg) if isinstance(arg, dict) else arg for arg in args)
        kwargs = {
            k: frozendict(v) if isinstance(v, dict) else v for k, v in kwargs.items()
        }
        return func(*args, **kwargs)

    return wrapped


@freezeargs
@lru_cache
def pick_calculator(
    library: Literal["fairchem", "matcalc", "rootstock"], **calc_kwargs: Any
) -> BaseCalculator:
    """
    Instantiate an ASE calculator from one of the supported MLIP libraries.

    Parameters
    ----------
    library
        MLIP library to use:
        - `fairchem` passes `**calc_kwargs` to `FAIRChemCalculator.from_model_checkpoint()`
        - `matcalc` passes `**calc_kwargs` to `matcalc.load_fp()`
        - `rootstock` passes `**calc_kwargs` to `rootstock.RootstockCalculator()`
    **calc_kwargs
        Keyword arguments forwarded to the loader function listed above for
        the given `library`. Every argument must be passed by keyword, even
        ones that are positional in the underlying function. Per library:

        - `fairchem`:
            - `name_or_path` (required): a model name from
              `fairchem.core.pretrained_mlip.available_models` or a path to a
              checkpoint file, e.g. `name_or_path="uma-s-1p1"`.
            - `task_name` (required for UMA checkpoints): one of `"omol"`,
              `"omat"`, `"oc20"`, `"odac"`, or `"omc"`.
            - Optional: `inference_settings`, `overrides`, `device`, and
              `workers`; see `FAIRChemCalculator.from_model_checkpoint` for
              details.
            - When Ray Serve batching is enabled, `source` can be set to
              `"path"` or `"registry"` to override automatic checkpoint
              resolution. `workers` values other than 1 and the deprecated
              `seed` argument are ignored with a warning because model loading
              happens in the Serve deployment.
            - Downloading gated checkpoints requires Hugging Face
              authentication.
        - `matcalc`:
            - `name` (required): the name of the foundation potential, e.g.
              `name="TensorNet-MatPES-PBE-2025.2"`. The supported names are
              the entries of `matcalc.UNIVERSAL_CALCULATOR_NAMES`
              (case-insensitive; common aliases are also accepted).
        - `rootstock`:
            - `checkpoint` (required): the checkpoint identifier, e.g.
              `checkpoint="mace-mp-0-medium"`. The corresponding environment
              must be pre-built via `rootstock install`.

    Returns
    -------
    BaseCalculator
        The instantiated calculator
    """

    from quacc import get_settings

    if has_torch:
        import torch

        cuda_is_available = torch.cuda.is_available()
    else:
        cuda_is_available = False

    settings = get_settings()
    library = library.lower()

    # Skip CUDA warning for rayserve batching mode (inference happens on remote GPU)
    use_ray_serve = library == "fairchem" and settings.FAIRCHEM_RAY_SERVE_BATCHING

    if not use_ray_serve and not cuda_is_available:
        LOGGER.warning("CUDA is not available to PyTorch. Calculations will be slow.")

    if library == "matcalc":
        from matcalc import __version__, load_fp

        calc = load_fp(**calc_kwargs)

    elif library == "rootstock":
        from rootstock import RootstockCalculator, __version__

        calc = RootstockCalculator(**calc_kwargs)
        # Enter the context manager to spawn the worker subprocess and load the
        # model. The lru_cache ensures __enter__ is called exactly once per
        # unique set of kwargs, keeping the worker warm across multiple calls.
        calc.__enter__()

    elif library == "fairchem":
        from fairchem.core import FAIRChemCalculator, __version__

        # Check if Ray Serve batching is enabled AND Ray is actually available
        if use_ray_serve:
            try:
                import ray

                if not ray.is_initialized():
                    LOGGER.warning(
                        "FAIRCHEM_RAY_SERVE_BATCHING is enabled but Ray is not initialized. "
                        "Falling back to local inference. To use Ray Serve batching, ensure "
                        "your flow is running with a Ray cluster (e.g., SlurmRayTaskRunner with "
                        "start_inference_server=True)."
                    )
                    use_ray_serve = False
            except ImportError:
                LOGGER.warning(
                    "FAIRCHEM_RAY_SERVE_BATCHING is enabled but Ray is not installed. "
                    "Falling back to local inference."
                )
                use_ray_serve = False

        if use_ray_serve:
            # Use Ray Serve multiplexed deployment for inference. The deployment
            # is expected to already be running on the cluster (typically started
            # by get_local_inference_raycluster / get_slurm_inference_raycluster with
            # setup_multiplexed_batch_predict_server). We connect by deployment
            # name and route requests to the appropriate model via a typed
            # ModelSpec. Its deterministic model_id includes all settings that
            # affect how the model is loaded.
            from fairchem.core.components.batch_server import ModelSpec
            from fairchem.core.units.mlip_unit.predict import BatchServerPredictUnit

            calc_kwargs = calc_kwargs.copy()  # Don't modify the original kwargs

            # Preserve the legacy checkpoint aliases and their precedence.
            name_or_path = calc_kwargs.pop("name_or_path", None)
            if name_or_path is not None:
                checkpoint = name_or_path
            else:
                checkpoint = calc_kwargs.pop("model_id", None) or calc_kwargs.pop(
                    "checkpoint", "uma-s-1p1"
                )

            task_name = calc_kwargs.pop("task_name")
            seed = calc_kwargs.pop("seed", None)
            workers = calc_kwargs.pop("workers", 1)
            if seed is not None:
                LOGGER.warning(
                    "The deprecated 'seed' argument is ignored when using FairChem "
                    "Ray Serve batching because the model is loaded by the deployment."
                )
            if workers != 1:
                LOGGER.warning(
                    "The 'workers' argument is ignored when using FairChem Ray Serve "
                    "batching; configure deployment replicas and GPU resources instead."
                )

            model_spec = ModelSpec(
                checkpoint=str(checkpoint),
                inference_settings=calc_kwargs.pop("inference_settings", "default"),
                device=calc_kwargs.pop("device", None),
                overrides=calc_kwargs.pop("overrides", None),
                source=calc_kwargs.pop("source", "auto"),
            )

            mlip_unit = BatchServerPredictUnit.from_deployment_connection_info(
                deployment_name="multiplexed-predict-server", model_spec=model_spec
            )

            calc = FAIRChemCalculator(predict_unit=mlip_unit, task_name=task_name)
        else:
            # Use local inference
            # ``source`` only disambiguates the multiplexed server loader;
            # local inference retains FAIRChem's name-or-existing-file lookup.
            calc_kwargs.pop("source", None)
            calc = FAIRChemCalculator.from_model_checkpoint(**calc_kwargs)

    else:
        raise ValueError(f"Unrecognized MLIP library {library}.")

    calc.parameters["version"] = __version__

    return calc
