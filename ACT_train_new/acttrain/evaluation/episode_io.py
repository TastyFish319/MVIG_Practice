"""Shared aligned whole-episode ACT inference, without hardware."""
from pathlib import Path
import numpy as np
from acttrain.model.inference import OBSERVATION_KEYS


def load_trace(root, entry):
    with np.load(root / entry["trace"], allow_pickle=False) as archive:
        state, target, full = (archive[key].copy() for key in ("state", "action", "full_state"))
        obs_ns, target_ns = (archive[key].copy() for key in ("timestamp_ns", "action_timestamp_ns"))
        action_source = archive["full_action_source"].copy() if "full_action_source" in archive else full[:, 7:14]
    rows = entry["rows"]
    if (rows < 1 or state.shape != (rows, 14) or target.shape != (rows, 7)
            or full.shape != (rows + 1, 14) or obs_ns.shape != (rows,) or target_ns.shape != (rows,)
            or not np.issubdtype(obs_ns.dtype, np.integer) or not np.issubdtype(target_ns.dtype, np.integer)
            or not all(np.isfinite(value).all() for value in (state, target, full))):
        raise ValueError("Invalid episode trace shapes, values or timestamps")
    if (not np.array_equal(state, full[:-1]) or action_source.shape != (rows + 1, 7)
            or not np.array_equal(target, action_source[1:])
            or np.any(target_ns - obs_ns != 50_000_000) or np.any(np.diff(obs_ns) != 50_000_000)
            or not np.array_equal(target_ns[:-1], obs_ns[1:])):
        raise ValueError("Episode trace violates next-frame action alignment / one-frame shift")
    return {"state": state, "target": target, "observation_timestamp_ns": obs_ns,
            "recorded_follower": full[1:, 7:14].copy(),
            "action_timestamp_ns": target_ns, "target_time_s": (target_ns - obs_ns[0]) / 1e9}


def predict_episode(runtime, dataset, offset, rows, execute_steps=1, batch_size=4,
                    expected_state=None, expected_episode_index=None):
    if rows < 1 or offset < 0 or not 1 <= execute_steps <= 32 or batch_size < 1:
        raise ValueError("Need nonempty rows, nonnegative offset, execute_steps in [1,32], batch_size >= 1")
    import torch

    queries = np.arange(0, rows, execute_steps, dtype=np.int64)
    chunks = []
    for begin in range(0, len(queries), batch_size):
        chosen = queries[begin:begin + batch_size]
        samples = [dataset[offset + int(row)] for row in chosen]
        for row, sample in zip(chosen, samples):
            if expected_episode_index is not None and int(sample["episode_index"]) != expected_episode_index:
                raise ValueError("Dataset/manifest episode indices differ")
            if expected_state is not None:
                np.testing.assert_allclose(sample["observation.state"].cpu().numpy(), expected_state[row],
                                           atol=1e-6, rtol=1e-5, err_msg="Dataset/trace state alignment differs")
        batch = {key: torch.stack([sample[key] for sample in samples]) for key in OBSERVATION_KEYS}
        if hasattr(runtime, "prepare_episode_batch"):
            batch = runtime.prepare_episode_batch(batch, expected_state, chosen)
        predicted = np.asarray(runtime.predict_batch(batch))
        if predicted.shape != (len(chosen), 32, 7) or not np.isfinite(predicted).all():
            raise ValueError("ACT returned invalid action chunks")
        chunks.append(predicted)
        print(f"Predicted {min(begin + batch_size, len(queries))}/{len(queries)} observation queries", flush=True)
    chunks = np.concatenate(chunks)
    origin = (np.arange(rows, dtype=np.int64) // execute_steps) * execute_steps
    horizon = np.arange(rows, dtype=np.int64) - origin
    # Last chunk is truncated at the selected episode boundary, never padded/scored beyond it.
    prediction = chunks[np.arange(rows) // execute_steps, horizon]
    return {"prediction": prediction, "prediction_chunks": chunks, "query_rows": queries,
            "prediction_origin_row": origin, "prediction_horizon_step": horizon}


