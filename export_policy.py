#!/usr/bin/env python3
"""Exports a Brax PPO checkpoint to a Pico-friendly C++ header.

This exporter is intentionally specific to the deployment observation used by
RickJoystickFlatTerrain. It restores the observation normalizer and policy
MLP, validates their shapes and semantics, and writes policy_weights.h.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


_LAYER_RE = re.compile(r"hidden_(\d+)$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Checkpoint step directory containing ppo_network_config.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("policy_weights.h"),
        help="Header to write (default: policy_weights.h beside this script)",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _restore_checkpoint(path: Path) -> list[Any]:
    try:
        import jax
        import orbax.checkpoint as ocp
    except ImportError as exc:
        raise RuntimeError(
            "Checkpoint export requires numpy, JAX, and orbax-checkpoint. "
            "Run this script from the mujoco_playground environment, or install "
            "them with: python -m pip install numpy jax orbax-checkpoint"
        ) from exc

    checkpointer = ocp.PyTreeCheckpointer()
    metadata = checkpointer.metadata(path).item_metadata
    restore_args = jax.tree.map(
        lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata
    )
    restored = checkpointer.restore(
        path,
        ocp.args.PyTreeRestore(restore_args=restore_args),
        item=None,
    )
    if not isinstance(restored, list) or len(restored) < 2:
        raise ValueError(
            "Expected a Brax PPO checkpoint containing normalizer, policy, and "
            f"value parameters; got {type(restored).__name__}"
        )
    return restored


def _environment_config_path(checkpoint: Path) -> Path:
    # Normal layout: <run>/checkpoints/<step>, with config.json in checkpoints.
    candidates = (
        checkpoint.parent / "config.json",
        checkpoint.parent.parent / "config.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find the environment config.json beside the checkpoint. "
        f"Checked: {', '.join(str(path) for path in candidates)}"
    )


def _float32_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must be floating point, got {array.dtype}")
    array = np.asarray(array, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _float_literal(value: float) -> str:
    value = float(np.float32(value))
    if not math.isfinite(value):
        raise ValueError(f"Cannot export non-finite value {value}")
    literal = format(value, ".9g")
    if "e" not in literal and "." not in literal:
        literal += ".0"
    return literal + "f"


def _format_array(name: str, values: np.ndarray) -> str:
    flat = np.ravel(values, order="C")
    lines = [f"static const float {name}[{flat.size}] = {{"]
    for start in range(0, flat.size, 8):
        row = ", ".join(_float_literal(value) for value in flat[start : start + 8])
        suffix = "," if start + 8 < flat.size else ""
        lines.append(f"    {row}{suffix}")
    lines.append("};")
    return "\n".join(lines)


def _ordered_layers(policy_params: dict[str, Any]) -> list[dict[str, Any]]:
    indexed_layers: list[tuple[int, dict[str, Any]]] = []
    for name, layer in policy_params.items():
        match = _LAYER_RE.fullmatch(name)
        if match:
            indexed_layers.append((int(match.group(1)), layer))
    indexed_layers.sort(key=lambda item: item[0])
    indices = [index for index, _ in indexed_layers]
    if indices != list(range(len(indices))):
        raise ValueError(f"Policy layers are not contiguous: {indices}")
    return [layer for _, layer in indexed_layers]


def _validate_configuration(
    network_config: dict[str, Any], env_config: dict[str, Any]
) -> tuple[int, int, list[int]]:
    observation_shape = network_config["observation_size"]["shape"]
    obs_dim = math.prod(int(size) for size in observation_shape)
    action_dim = int(network_config["action_size"])
    kwargs = network_config["network_factory_kwargs"]
    hidden_dims = [int(size) for size in kwargs["policy_hidden_layer_sizes"]]

    if not network_config.get("normalize_observations", False):
        raise ValueError("Firmware export requires normalized observations")
    if kwargs.get("activation") not in ("silu", "swish"):
        raise ValueError(
            "Firmware implements SiLU/Swish, but the checkpoint activation is "
            f"{kwargs.get('activation')!r}"
        )
    if kwargs.get("distribution_type") != "tanh_normal":
        raise ValueError(
            "Firmware implements deterministic tanh-normal actions, but the "
            f"checkpoint distribution is {kwargs.get('distribution_type')!r}"
        )
    if kwargs.get("policy_obs_key", "state") != "state":
        raise ValueError("Only the PPO 'state' policy observation is supported")

    history_length = int(env_config["command_history_length"])
    expected_obs_dim = history_length * action_dim + 3 + 3 + 2 + 1
    if obs_dim != expected_obs_dim:
        raise ValueError(
            "Rick deployment observation mismatch: expected "
            f"{history_length} * {action_dim} + 9 = {expected_obs_dim}, got {obs_dim}"
        )
    return obs_dim, action_dim, hidden_dims


def _numpy_policy(
    observations: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    kernels: list[np.ndarray],
    biases: list[np.ndarray],
    action_dim: int,
) -> np.ndarray:
    activations = (np.asarray(observations, dtype=np.float32) - mean) / std
    for index, (kernel, bias) in enumerate(zip(kernels, biases, strict=True)):
        activations = activations @ kernel + bias
        if index + 1 < len(kernels):
            sigmoid = np.empty_like(activations)
            positive = activations >= 0.0
            sigmoid[positive] = 1.0 / (1.0 + np.exp(-activations[positive]))
            exp_x = np.exp(activations[~positive])
            sigmoid[~positive] = exp_x / (1.0 + exp_x)
            activations = activations * sigmoid
    return np.tanh(activations[..., :action_dim])


def _build_header(checkpoint: Path) -> tuple[str, int, float]:
    network_config = _load_json(checkpoint / "ppo_network_config.json")
    env_config = _load_json(_environment_config_path(checkpoint))
    obs_dim, action_dim, hidden_dims = _validate_configuration(
        network_config, env_config
    )

    restored = _restore_checkpoint(checkpoint)
    normalizer = restored[0]
    policy_tree = restored[1]
    if not isinstance(normalizer, dict) or not isinstance(policy_tree, dict):
        raise ValueError("Unexpected Brax checkpoint tree structure")

    obs_mean = _float32_array(normalizer["mean"], "observation mean")
    obs_std = _float32_array(normalizer["std"], "observation standard deviation")
    if obs_mean.shape != (obs_dim,) or obs_std.shape != (obs_dim,):
        raise ValueError(
            f"Normalizer shapes must both be ({obs_dim},), got "
            f"{obs_mean.shape} and {obs_std.shape}"
        )
    if np.any(obs_std <= 0.0):
        raise ValueError("Observation standard deviations must be positive")

    layers = _ordered_layers(policy_tree["params"])
    expected_out_dims = hidden_dims + [2 * action_dim]
    if len(layers) != len(expected_out_dims):
        raise ValueError(
            f"Expected {len(expected_out_dims)} dense layers, got {len(layers)}"
        )

    kernels: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    in_dim = obs_dim
    for index, (layer, out_dim) in enumerate(
        zip(layers, expected_out_dims, strict=True)
    ):
        kernel = _float32_array(layer["kernel"], f"layer {index} kernel")
        bias = _float32_array(layer["bias"], f"layer {index} bias")
        if kernel.shape != (in_dim, out_dim) or bias.shape != (out_dim,):
            raise ValueError(
                f"Layer {index} expected kernel {(in_dim, out_dim)} and bias "
                f"{(out_dim,)}, got {kernel.shape} and {bias.shape}"
            )
        kernels.append(kernel)
        biases.append(bias)
        in_dim = out_dim

    test_observations = np.stack(
        (
            np.zeros(obs_dim, dtype=np.float32),
            obs_mean,
            np.linspace(-0.75, 0.75, obs_dim, dtype=np.float32),
        )
    )
    reference_actions = _numpy_policy(
        test_observations, obs_mean, obs_std, kernels, biases, action_dim
    )

    source = checkpoint.resolve().as_posix()
    checkpoint_step = int(checkpoint.name)
    parameter_count = sum(
        kernel.size + bias.size
        for kernel, bias in zip(kernels, biases, strict=True)
    )
    max_layer_dim = max(expected_out_dims)
    in_dims = [obs_dim] + expected_out_dims[:-1]

    sections = [
        "// Generated by export_policy.py. Do not edit by hand.",
        f"// Source checkpoint: {source}",
        f"// Policy parameters: {parameter_count} float32 values",
        "#ifndef RICK_POLICY_WEIGHTS_H",
        "#define RICK_POLICY_WEIGHTS_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define POLICY_OBS_DIM {obs_dim}",
        f"#define POLICY_ACTION_DIM {action_dim}",
        f"#define POLICY_OUTPUT_DIM {2 * action_dim}",
        f"#define POLICY_DENSE_LAYER_COUNT {len(layers)}",
        f"#define POLICY_MAX_LAYER_DIM {max_layer_dim}",
        f"#define POLICY_COMMAND_HISTORY_LENGTH {int(env_config['command_history_length'])}",
        f"#define POLICY_CHECKPOINT_STEP {checkpoint_step}",
        "",
        f"static const float POLICY_CONTROL_DT = {_float_literal(env_config['ctrl_dt'])};",
        f"static const float POLICY_STEP_FREQUENCY = {_float_literal(env_config['step_frequency'])};",
        f"static const float POLICY_GYRO_OBS_SCALE = {_float_literal(env_config['gyro_obs_scale'])};",
        f"static const float POLICY_TARGET_VELOCITY = {_float_literal(env_config['target_velocity'])};",
        f"static const float POLICY_ACTION_SCALE_RAD = {_float_literal(env_config['action_scale'])};",
        "",
        _format_array("POLICY_OBS_MEAN", obs_mean),
        "",
        _format_array("POLICY_OBS_STD", obs_std),
    ]
    for index, (kernel, bias) in enumerate(
        zip(kernels, biases, strict=True)
    ):
        sections.extend(
            (
                "",
                _format_array(f"POLICY_LAYER_{index}_KERNEL", kernel),
                "",
                _format_array(f"POLICY_LAYER_{index}_BIAS", bias),
            )
        )

    kernel_names = ",\n    ".join(
        f"POLICY_LAYER_{index}_KERNEL" for index in range(len(layers))
    )
    bias_names = ",\n    ".join(
        f"POLICY_LAYER_{index}_BIAS" for index in range(len(layers))
    )
    sections.extend(
        (
            "",
            "static const float *const POLICY_LAYER_KERNELS[POLICY_DENSE_LAYER_COUNT] = {\n"
            f"    {kernel_names}\n"
            "};",
            "",
            "static const float *const POLICY_LAYER_BIASES[POLICY_DENSE_LAYER_COUNT] = {\n"
            f"    {bias_names}\n"
            "};",
            "",
            "static const uint16_t POLICY_LAYER_IN_DIMS[POLICY_DENSE_LAYER_COUNT] = {",
            "    " + ", ".join(str(value) for value in in_dims),
            "};",
            "",
            "static const uint16_t POLICY_LAYER_OUT_DIMS[POLICY_DENSE_LAYER_COUNT] = {",
            "    " + ", ".join(str(value) for value in expected_out_dims),
            "};",
            "",
            "#endif  // RICK_POLICY_WEIGHTS_H",
            "",
        )
    )

    # Nine significant digits round-trip every IEEE-754 float32. Confirm the
    # generated literals preserve the reference policy before writing them.
    quantized_mean = np.array(
        [np.float32(float(_float_literal(value)[:-1])) for value in obs_mean]
    )
    quantized_std = np.array(
        [np.float32(float(_float_literal(value)[:-1])) for value in obs_std]
    )
    quantized_kernels = [
        np.array(
            [np.float32(float(_float_literal(value)[:-1])) for value in kernel.flat],
            dtype=np.float32,
        ).reshape(kernel.shape)
        for kernel in kernels
    ]
    quantized_biases = [
        np.array(
            [np.float32(float(_float_literal(value)[:-1])) for value in bias.flat],
            dtype=np.float32,
        )
        for bias in biases
    ]
    exported_actions = _numpy_policy(
        test_observations,
        quantized_mean,
        quantized_std,
        quantized_kernels,
        quantized_biases,
        action_dim,
    )
    max_action_error = float(np.max(np.abs(reference_actions - exported_actions)))
    if max_action_error > 1e-6:
        raise ValueError(
            f"Generated float literals changed policy output by {max_action_error}"
        )
    return "\n".join(sections), parameter_count, max_action_error


def main() -> None:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_dir():
        raise NotADirectoryError(f"Checkpoint directory not found: {checkpoint}")

    header, parameter_count, max_action_error = _build_header(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_output.write_text(header, encoding="utf-8")
    temporary_output.replace(output)
    print(f"Wrote {output}")
    print(f"Exported {parameter_count} policy parameters")
    print(f"Float32 round-trip max action error: {max_action_error:.3g}")


if __name__ == "__main__":
    main()
