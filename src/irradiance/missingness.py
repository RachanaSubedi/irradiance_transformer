"""Training-time missingness augmentation for irradiance imputation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


POINT = 0
BLOCK = 1
WHOLE_TARGET = 2


@dataclass(frozen=True)
class MissingnessConfig:
    """Probabilities control target gap type; source dropout is independent."""

    point_mode_probability: float = 0.20
    block_mode_probability: float = 0.30
    whole_target_probability: float = 0.50
    point_mask_probability: float = 0.10
    block_lengths_steps: tuple[int, ...] = (6, 12, 24, 36, 48, 72)
    source_dropout_probability: float = 0.15
    elapsed_cap_minutes: float = 1440.0

    def __post_init__(self) -> None:
        probabilities = (
            self.point_mode_probability,
            self.block_mode_probability,
            self.whole_target_probability,
        )
        if any(value < 0 for value in probabilities):
            raise ValueError("mode probabilities must be nonnegative")
        if abs(sum(probabilities) - 1.0) > 1e-8:
            raise ValueError("target missingness probabilities must sum to one")
        if not 0 <= self.point_mask_probability <= 1:
            raise ValueError("point_mask_probability must lie in [0,1]")
        if not 0 <= self.source_dropout_probability <= 1:
            raise ValueError("source_dropout_probability must lie in [0,1]")
        if not self.block_lengths_steps or any(x <= 0 for x in self.block_lengths_steps):
            raise ValueError("block lengths must be positive")
        if self.elapsed_cap_minutes <= 0:
            raise ValueError("elapsed cap must be positive")


class MissingnessCurriculum:
    """Apply artificial target masks to a collated CPU batch.

    The original target values and natural truth masks are never modified.
    Returned ``imputation_loss_mask`` marks only naturally observed values that
    were deliberately hidden. Source-station dropout is sampled independently.
    """

    def __init__(
        self,
        config: MissingnessConfig = MissingnessConfig(),
        *,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def _sample_modes(self, batch_size: int) -> torch.Tensor:
        probabilities = torch.tensor(
            [
                self.config.point_mode_probability,
                self.config.block_mode_probability,
                self.config.whole_target_probability,
            ],
            dtype=torch.float32,
        )
        return torch.multinomial(
            probabilities,
            batch_size,
            replacement=True,
            generator=self.generator,
        )

    def _target_mask(
        self,
        truth_mask: torch.Tensor,
        center: int,
        modes: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length = truth_mask.shape
        artificial = torch.zeros_like(truth_mask, dtype=torch.bool)

        for row in range(batch_size):
            mode = int(modes[row])
            if mode == POINT:
                sampled = torch.rand(
                    sequence_length, generator=self.generator
                ) < self.config.point_mask_probability
                artificial[row] = sampled & truth_mask[row]
                artificial[row, center] = truth_mask[row, center]

            elif mode == BLOCK:
                valid_lengths = [
                    length for length in self.config.block_lengths_steps
                    if length <= sequence_length
                ]
                if not valid_lengths:
                    raise ValueError("no block length fits the sequence")
                choice = int(
                    torch.randint(
                        len(valid_lengths), (1,), generator=self.generator
                    )
                )
                length = valid_lengths[choice]
                earliest = max(0, center - length + 1)
                latest = min(center, sequence_length - length)
                start = int(
                    torch.randint(
                        earliest,
                        latest + 1,
                        (1,),
                        generator=self.generator,
                    )
                )
                artificial[row, start : start + length] = True
                artificial[row] &= truth_mask[row]

            elif mode == WHOLE_TARGET:
                artificial[row] = truth_mask[row]
            else:
                raise AssertionError(f"unknown mask mode {mode}")

        return artificial

    @staticmethod
    def _recompute_elapsed(
        observed_mask: torch.Tensor,
        original_elapsed: torch.Tensor,
        whole_target_rows: torch.Tensor,
        target_indices: torch.Tensor,
        cap_minutes: float,
    ) -> torch.Tensor:
        result = original_elapsed.clone()
        batch_size, sequence_length, stations = observed_mask.shape

        for row in range(batch_size):
            target = int(target_indices[row])
            if bool(whole_target_rows[row]):
                result[row, :, target] = cap_minutes
                continue

            elapsed = float(result[row, 0, target])
            for step in range(sequence_length):
                if bool(observed_mask[row, step, target]):
                    elapsed = 0.0
                else:
                    elapsed = min(elapsed + 5.0, cap_minutes)
                result[row, step, target] = elapsed
        return result

    def _drop_sources(
        self,
        source_station_mask: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        allowed = source_station_mask.clone().bool()
        dropped = torch.zeros_like(allowed)

        for row in range(len(allowed)):
            candidates = torch.nonzero(
                allowed[row], as_tuple=False
            ).flatten()
            if len(candidates) <= 1:
                continue
            decisions = torch.rand(
                len(candidates), generator=self.generator
            ) < self.config.source_dropout_probability

            # Retain at least one permitted source station.
            if bool(decisions.all()):
                keep_position = int(
                    torch.randint(
                        len(candidates), (1,), generator=self.generator
                    )
                )
                decisions[keep_position] = False
            chosen = candidates[decisions]
            dropped[row, chosen] = True
            allowed[row, chosen] = False

        row = torch.arange(len(allowed))
        allowed[row, target_indices] = False
        return allowed, dropped

    def __call__(self, batch: dict[str, torch.Tensor], *, center: int) -> dict[str, torch.Tensor]:
        required = {
            "station_csi",
            "station_csi_mask",
            "time_since_observed_minutes",
            "meteorology",
            "meteorology_mask",
            "target_csi_sequence",
            "target_truth_mask",
            "target_station_index",
            "source_station_mask",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(f"batch is missing fields {missing}")

        # Augmentation is intentionally performed on CPU before device transfer.
        if batch["station_csi"].device.type != "cpu":
            raise ValueError("apply MissingnessCurriculum before moving batch to GPU")

        output = dict(batch)
        csi = batch["station_csi"].clone()
        observed = batch["station_csi_mask"].clone().bool()
        elapsed = batch["time_since_observed_minutes"].clone()
        meteorology = batch["meteorology"].clone()
        meteorology_mask = batch["meteorology_mask"].clone().bool()
        target = batch["target_station_index"].long()
        truth_mask = batch["target_truth_mask"].bool()
        batch_size, sequence_length, _ = csi.shape

        if not 0 <= center < sequence_length:
            raise ValueError("center lies outside sequence")
        row = torch.arange(batch_size)
        if not bool(truth_mask[:, center].all()):
            raise ValueError("every training sample must have center truth")
        # Detect accidental double masking from whole_target_blackout=True.
        if not bool(observed[row, center, target].all()):
            raise ValueError(
                "target center is already hidden; construct the dataset with "
                "whole_target_blackout=False before applying the curriculum"
            )

        modes = self._sample_modes(batch_size)
        artificial_target = self._target_mask(truth_mask, center, modes)

        for sample in range(batch_size):
            station = int(target[sample])
            positions = artificial_target[sample]
            csi[sample, positions, station] = 0.0
            observed[sample, positions, station] = False
            # We simulate station/device outages, so target-site meteorology is
            # hidden at the same artificially missing timestamps.
            meteorology[sample, positions, station] = 0.0
            meteorology_mask[sample, positions, station] = False

        source_allowed, dropped_sources = self._drop_sources(
            batch["source_station_mask"], target
        )
        for sample in range(batch_size):
            for station in torch.nonzero(
                dropped_sources[sample], as_tuple=False
            ).flatten():
                station = int(station)
                csi[sample, :, station] = 0.0
                observed[sample, :, station] = False
                elapsed[sample, :, station] = self.config.elapsed_cap_minutes
                meteorology[sample, :, station] = 0.0
                meteorology_mask[sample, :, station] = False

        elapsed = self._recompute_elapsed(
            observed,
            elapsed,
            modes == WHOLE_TARGET,
            target,
            self.config.elapsed_cap_minutes,
        )

        loss_mask = artificial_target & truth_mask
        if not bool(loss_mask[:, center].all()):
            raise AssertionError("center truth must always be artificially hidden")

        artificial_full = torch.zeros_like(observed)
        artificial_full[row[:, None], torch.arange(sequence_length)[None, :], target[:, None]] = artificial_target

        output.update(
            {
                "station_csi": csi,
                "station_csi_mask": observed,
                "time_since_observed_minutes": elapsed,
                "meteorology": meteorology,
                "meteorology_mask": meteorology_mask,
                "source_station_mask": source_allowed,
                "artificial_missing_mask": artificial_full,
                "imputation_loss_mask": loss_mask,
                "missingness_mode": modes,
                "dropped_source_mask": dropped_sources,
            }
        )
        return output
