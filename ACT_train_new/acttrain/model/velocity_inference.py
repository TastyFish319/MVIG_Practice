"""Checkpoint-safe v3 inference with explicit episode history."""
from acttrain.model.inference import ACTInference, checkpoint_directory
from acttrain.data.core import read_json
from acttrain.model.velocity import CONTRACT, CONTRACT_FILE, prepare_batch


class VelocityInference(ACTInference):
    state_dim = 21

    def __init__(self, checkpoint, device="cuda", expected_stats=None):
        directory = checkpoint_directory(checkpoint)
        contract = read_json(directory / CONTRACT_FILE)
        if any(contract.get(k) != v for k, v in CONTRACT.items()):
            raise ValueError("Unsupported ACT v3 observation contract")
        from acttrain.model.relative import checkpoint_contract
        action = checkpoint_contract(directory)
        if action is None or action.get("dataset_manifest_sha256") != contract.get("dataset_manifest_sha256"):
            raise ValueError("V3 requires matching observation and relative-action contracts")
        self.observation_contract = contract
        stats = {**(expected_stats or {}), "observation.state": contract["state_stats"], "action": action["action_stats"]}
        super().__init__(directory, device, expected_stats=stats)

    @staticmethod
    def prepare_episode_batch(batch, states, indices):
        if states is None:
            raise ValueError("V3 requires explicit recorded episode history")
        return prepare_batch(batch, states, indices)

    def predict_rgb(self, state, left_rgb, right_rgb):
        raise ValueError("V3 requires history: use prepare_episode_batch and predict_batch; state14-only live serving is unsupported")
