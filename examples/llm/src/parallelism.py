# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for MegaBlocks expert model parallelism."""

from composer.utils import dist
from omegaconf import DictConfig
import torch


def _expert_parallel_group_size(cfg: DictConfig):
    # Get the expert parallel group size from the config. If it isn't specified,
    # use expert parallelism across the maximum number of devices possible.
    num_experts = cfg.moe.num_experts
    expert_parallel_group_size = (
        cfg.moe.get('expert_model_parallel_group_size', None))
    expert_parallel_group_size = (
        expert_parallel_group_size or min(dist.get_world_size(), num_experts))
    if (num_experts % expert_parallel_group_size) != 0:
        raise ValueError(
            "The number of experts is not divisible by the number of devices. "
            "Please specify an explicit `expert_parallel_dimension` that is "
            f"divisible by {dist.get_world_size()}.")
    return expert_parallel_group_size


def create_moe_expert_parallel_group(cfg: DictConfig):
    if not cfg.get('moe', None):
        raise ValueError(
            "Cannot create MoE expert parallel group with "
            "no MoE arguments in 'cfg'.")

    # NOTE: We map consecutive ranks to the same expert parallel group.
    rank = dist.get_global_rank()
    expert_parallel_group_size = _expert_parallel_group_size(cfg)
    expert_parallel_group_index = rank // expert_parallel_group_size

    start_rank = expert_parallel_group_index * expert_parallel_group_size
    end_rank = start_rank + expert_parallel_group_size
    ranks = list(range(start_rank, end_rank))
    return torch.distributed.new_group(ranks=ranks)


def create_moe_data_parallel_group(cfg: DictConfig):
    if not cfg.get('moe', None):
        raise ValueError(
            "Cannot create MoE data parallel group with "
            "no MoE arguments in 'cfg'.")

    rank = dist.get_global_rank()
    expert_parallel_group_size = _expert_parallel_group_size(cfg)
    data_parallel_group_index = rank % expert_parallel_group_size

    # NOTE: Group ranks across the expert parallel groups.
    ranks = list(range(
        data_parallel_group_index,
        dist.get_world_size(),
        expert_parallel_group_size))
    return torch.distributed.new_group(ranks=ranks)
