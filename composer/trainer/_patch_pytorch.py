# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Released under BSD 3-Clause License,
# Copyright (c) Facebook, Inc. and its affiliates.

# yapf: disable
# isort: skip_file
# pyright: reportGeneralTypeIssues=false

"""PyTorch, especially PyTorch Distributed, monkeypatches."""

import logging
import math
from typing import Any, Iterable, Optional, Union, no_type_check

import torch
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.distributed._shard.sharding_spec import ShardMetadata
from torch.distributed._shard.sharding_spec._internals import get_chunked_dim_size, get_split_size
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy
from torch.distributed.fsdp._fsdp_extensions import _ext_pre_load_state_dict_transform
from torch.distributed.utils import _replace_by_prefix

log = logging.getLogger(__name__)


def patch_pytorch():
    """Monkey patches pytorch functions based on pytorch version."""
    if version.parse(torch.__version__) < version.parse('2.1.1'):
        # Monkey patch for torch < 2.1.1 ie torch == 2.1.0

        # Monkey patch sharding method
        ChunkShardingSpec.build_metadata = build_metadata

        # Monkey patch partial state dict handling
        from torch.distributed.fsdp import _state_dict_utils

        _state_dict_utils._sharded_pre_load_state_dict_hook = (_sharded_pre_load_state_dict_hook)

        # Allow 2D HSDP
        from torch.distributed.fsdp import _runtime_utils
        _runtime_utils._validate_and_get_hybrid_shard_state = lambda *args, **kwargs: None

    elif version.parse(torch.__version__) < version.parse('2.1.3'):
        # Monkey patch for torch < 2.1.3 ie torch == 2.1.1, 2.1.2

        # Allow 2D HSDP
        from torch.distributed.fsdp import _runtime_utils
        _runtime_utils._validate_and_get_hybrid_shard_state = lambda *args, **kwargs: None

    elif version.parse(torch.__version__) < version.parse('2.2.1'):
        # Monkey patch for torch < 2.2.1 ie torch == 2.2.0

        # Allow 2D HSDP
        from torch.distributed.fsdp import _runtime_utils
        _runtime_utils._validate_and_get_hybrid_shard_state = lambda *args, **kwargs: None

    elif version.parse(torch.__version__) < version.parse('2.2.3'):
        # Monkey patch for torch < 2.2.3 ie torch == 2.2.1/2.2.2 currently

        # Fix memory leak for FSDP.optim_state_dict_to_load
        # https://github.com/pytorch/pytorch/issues/116553
        from torch.distributed.fsdp import _optim_utils

        _optim_utils._shard_orig_param_state = _shard_orig_param_state

    elif version.parse(torch.__version__) < version.parse('2.3.1'):
        # Monkey patch for torch < 2.3.1 ie torch == 2.3.0

        # Monkeypatch _flat_param.py to fix 2D with SHARD_GRAD_OP
        # Issue: https://github.com/pytorch/pytorch/issues/123272
        from torch.distributed.fsdp import _flat_param

        _flat_param._same_storage = _same_storage

        # Monkeypatch state_dict to get FQNs correctly.
        # Issue: https://github.com/pytorch/pytorch/pull/124698
        from torch.distributed.checkpoint import state_dict

        state_dict.set_model_state_dict = set_model_state_dict
        state_dict.set_optimizer_state_dict = set_optimizer_state_dict
        state_dict._get_fqns = _get_fqns

        # Monkeypatch for ND child submeshes
        # PR: https://github.com/pytorch/pytorch/pull/119752
        from torch.distributed.device_mesh import DeviceMesh, _MeshEnv

        _MeshEnv.create_child_mesh = create_child_mesh
        DeviceMesh.__getitem__ = device_mesh__getitem__
        DeviceMesh.__init__ = device_mesh__init__


def build_metadata(
    self,
    tensor_sizes: torch.Size,
    tensor_properties: sharded_tensor_meta.TensorProperties,
) -> sharded_tensor_meta.ShardedTensorMetadata:
    """Adds nightly change for ChunkShardingSpec.

    Change implemented in https://github.com/pytorch/pytorch/pull/108915
    """
    tensor_num_dim = len(tensor_sizes)

    self._verify_dim(self.dim)
    if self.dim >= tensor_num_dim or self.dim < -tensor_num_dim:  # type: ignore[operator]
        raise ValueError(f'Invalid sharding dim: {self.dim}')

    shards_metadata = []
    sharding_dim_size = tensor_sizes[self.dim]  # type: ignore[index]
    chunks = len(self.placements)
    split_size = get_split_size(sharding_dim_size, chunks)
    for idx, placement in enumerate(self.placements):
        # generate ShardMetadata for each placement device
        chunked_dim_size = get_chunked_dim_size(sharding_dim_size, split_size, idx)
        shard_size = list(tensor_sizes)
        current_offsets = [0] * tensor_num_dim
        current_offsets[self.dim] = split_size * idx  # type: ignore[index]
        shard_size[self.dim] = chunked_dim_size  # type: ignore[index]

        shard_metadata = ShardMetadata(
            shard_offsets=current_offsets,
            shard_sizes=shard_size,
            placement=placement,
        )
        shards_metadata.append(shard_metadata)

    return sharded_tensor_meta.ShardedTensorMetadata(shards_metadata, tensor_sizes, tensor_properties)


@no_type_check
def _sharded_pre_load_state_dict_hook(
    module: nn.Module,
    fsdp_state,
    state_dict: dict[str, Any],
    prefix: str,
) -> None:
    """Adds nightly change for partial state dict error handling.

    https://github.com/pytorch/pytorch/blob/0511df0ee9edeb5c2613805ccfb49beb323b87f9/torch/distributed/fsdp/_state_dict_utils.py#L607-L615

    The hook combines the unflattened, sharded parameters (ShardedTensor) to
    a new FlatParameter and shards the new FlatParameter to the local chunk.
    """
    from torch.distributed._tensor import Replicate
    from torch.distributed.distributed_c10d import _get_pg_default_device
    from torch.distributed.fsdp._common_utils import FSDP_PREFIX, _has_fsdp_params, _is_composable, _module_handle
    from torch.distributed.fsdp._runtime_utils import _lazy_init
    from torch.distributed.fsdp._state_dict_utils import _enter_unshard_params_ctx, _param_name_infos

    _lazy_init(fsdp_state, module)
    if not _is_composable(fsdp_state):
        _replace_by_prefix(state_dict, prefix, prefix + f'{FSDP_PREFIX}')
    if not _has_fsdp_params(fsdp_state, module):
        return

    handle = _module_handle(fsdp_state, module)
    if not handle.uses_sharded_strategy:  # type: ignore
        raise RuntimeError(
            'load_sharded_state_dict can only be called when parameters '
            'are flattened and sharded.',
        )

    device = fsdp_state.compute_device
    for fqn, _, _ in _param_name_infos(module, fsdp_state):
        if not _is_composable(fsdp_state):
            fqn_from_global_root = f'{prefix}{FSDP_PREFIX}{fqn}'
        else:
            fqn_from_global_root = f'{prefix}{fqn}'
        try:
            param = state_dict.pop(fqn_from_global_root)
        except KeyError:
            log.warning(
                f'Did not find param with FQN {fqn_from_global_root}, skipping it. '  # noqa: G004
                'The weight will not be filled if you expect it to be.',
            )
            continue  # TODO: Improve unittesting for state_dict finetuning
            # cases: https://github.com/pytorch/pytorch/issues/109134

        if not fsdp_state._state_dict_config.use_dtensor:
            # All-gather the param (ShardedTensor)
            param, shards = _ext_pre_load_state_dict_transform(param)

            assert len(shards) < 2, (
                'Expects 0 or 1 shard per rank '
                f'but got {len(shards)} shards on rank {fsdp_state.rank}.'
            )
            param_numel = param.size().numel()
            dim_0_size = param.size()[0]
            chunk_size = (math.ceil(dim_0_size / fsdp_state.world_size) * param_numel // dim_0_size)
            if len(shards) == 1:
                local_tensor = shards[0].tensor.flatten()
                pg_device = _get_pg_default_device(fsdp_state.process_group)
                if local_tensor.device.type != pg_device.type:
                    local_tensor = local_tensor.to(pg_device)
                num_padding = chunk_size - local_tensor.numel()
                if num_padding > 0:
                    local_tensor = F.pad(local_tensor, [0, num_padding])
            else:
                local_tensor = torch.zeros(chunk_size, dtype=param.dtype, device=device)
            tensor = torch.empty(
                chunk_size * fsdp_state.world_size,
                dtype=local_tensor.dtype,
                device=device,
            )
            if local_tensor.is_cpu:
                # Tensor could be on FSDP GPU compute device, while local_tensor is on CPU.
                # Convert to CPU so all_gather can work.
                tensor_dev = tensor.device
                tensor = tensor.cpu()
                tensor_list = list(torch.chunk(tensor, torch.distributed.get_world_size(fsdp_state.process_group)))
                torch.distributed.all_gather(tensor_list, local_tensor, group=fsdp_state.process_group)
                tensor.to(tensor_dev)
            else:
                torch.distributed.all_gather_into_tensor(tensor, local_tensor, group=fsdp_state.process_group)
            tensor = tensor.narrow(0, 0, param_numel).reshape(param.size())
            state_dict[fqn_from_global_root] = tensor
        else:
            if param.device != fsdp_state._device_mesh.device_type:  # type: ignore
                param = param.to(fsdp_state._device_mesh.device_type)  # type: ignore

            param = param.redistribute(device_mesh=param.device_mesh, placements=[Replicate()])
            state_dict[fqn_from_global_root] = param.to_local()

    _enter_unshard_params_ctx(module, fsdp_state, writeback=True)


if version.parse(torch.__version__) >= version.parse('2.2.1') and version.parse(
        torch.__version__,) < version.parse('2.2.3'):

    from torch.distributed.fsdp._optim_utils import FSDPParamInfo
    from torch.distributed.checkpoint._state_dict_utils import _gather_state_dict

    @no_type_check
    def _shard_orig_param_state(
        fsdp_param_info: FSDPParamInfo,
        fqn: str,
        optim_state: dict[str, Any],
    ) -> dict[str, Any]:
        if not optim_state:
            return {}
        fsdp_state = fsdp_param_info.state
        flat_param = fsdp_param_info.handle.flat_param
        param_idx = fsdp_param_info.param_indices[fqn]
        shard_param_info = flat_param._shard_param_infos[param_idx]  # type: ignore[attr-defined]
        optim_state = _gather_state_dict(
            optim_state, pg=fsdp_state.process_group, device=fsdp_state.compute_device,
        )
        if not shard_param_info.in_shard:
            return {}
        # Flatten and shard the state.
        new_optim_state: dict[str, Any] = {}
        intra_param_start_idx = shard_param_info.intra_param_start_idx
        intra_param_end_idx = shard_param_info.intra_param_end_idx
        for state_name, value in optim_state.items():
            if (
                torch.is_tensor(value)
                and value.dim() > 0
                and fsdp_state.sharding_strategy != ShardingStrategy.NO_SHARD
            ):
                # This clone() is the patch to fix the OOM
                # https://github.com/pytorch/pytorch/pull/117261
                value = value.flatten()[intra_param_start_idx : intra_param_end_idx + 1].clone()  # type: ignore[operator]
            new_optim_state[state_name] = value
        return new_optim_state


if version.parse(torch.__version__) >= version.parse('2.3.0') and version.parse(
        torch.__version__,
) < version.parse('2.3.1'):
    from torch.distributed._tensor import DTensor

    @no_type_check
    def _same_storage(a, b):
        if isinstance(a, DTensor):
            a = a._local_tensor
        if isinstance(b, DTensor):
            b = b._local_tensor
        return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()

    from torch.distributed.checkpoint.state_dict import (_unflatten_model_state_dict, _verify_options,
                                                         _load_model_state_dict, gc_context,
                                                         _verify_state_dict, _load_optim_state_dict,
                                                         FQNS_T)

    @no_type_check
    def _get_fqns(
        model: nn.Module,
        name: str,
        skip_ddp_prefix: bool = True,
        skip_compiler_prefix: bool = True,
    ) -> FQNS_T:
        """Used to convert the name of a parameter to the FQNs.

        For FSDP without `use_orig_params`, the name of FlatParameter can be mapped to
        multiple original parameters. As a result, the return type of this function
        is `set[str]`.

        Args:
            module (nn.Module): the root model.
            name (str): the name
            skip_ddp_prefix (bool): whether to skip DDP's `module` prefix

        Returns:
            The canonical FQNs based on the model traversal.
        """
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import _CHECKPOINT_PREFIX
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.checkpoint.state_dict import FLAT_PARAM
        from torch.distributed.fsdp._common_utils import FSDP_WRAPPED_MODULE

        # Remove the checkpoint prefix, if it exists.
        name = name.replace(_CHECKPOINT_PREFIX, '')
        if '.' not in name:
            return {name}

        obj_names = name.split('.')
        fqn_obj_names = []
        curr_obj = model
        for i, curr_obj_name in enumerate(obj_names):
            if isinstance(curr_obj, DDP):
                assert curr_obj_name == 'module'
                curr_obj = curr_obj.module
                if not skip_ddp_prefix:
                    fqn_obj_names.append(curr_obj_name)
            elif isinstance(curr_obj, FSDP):
                if i < len(obj_names) - 1 and obj_names[i + 1] == FLAT_PARAM:
                    prefix = '.'.join(fqn_obj_names)
                    flat_param = getattr(curr_obj, FLAT_PARAM)
                    if prefix:
                        prefix = f'{prefix}.'
                    return {f'{prefix}{fqn}' for fqn in flat_param._fqns}
                curr_obj = getattr(curr_obj, FSDP_WRAPPED_MODULE)
                if curr_obj_name != FSDP_WRAPPED_MODULE:
                    fqn_obj_names.append(curr_obj_name)
                    curr_obj = getattr(curr_obj, curr_obj_name)
            elif isinstance(curr_obj, torch._dynamo.eval_frame.OptimizedModule):
                assert curr_obj_name == '_orig_mod'
                curr_obj = curr_obj._orig_mod
                if not skip_compiler_prefix:
                    fqn_obj_names.append(curr_obj_name)
            else:
                fqn_obj_names.append(curr_obj_name)
                curr_obj = getattr(curr_obj, curr_obj_name)

        return {'.'.join(fqn_obj_names).replace(_CHECKPOINT_PREFIX, '')}

    def set_model_state_dict(
        model: nn.Module,
        model_state_dict,
        *,
        options = None,
    ):
        """Load the model state_dict.

        The counterpart of ``get_model_state_dict`` to set the state_dict to the
        model. See ``set_state_dict`` for the detail usage.

        Args:
            model (nn.Module): the nn.Module to the model.
            model_state_dict: (dict[str, ValueType]):
            the model state_dict to load. If the key of the ``model_state_dict``
            is nn.Module, the key is a submodule of ``model`` and the value should
            be the state_dict of the submodule. When loading the state_dict,
            the prefix of the submodule will be append to the state_dict.
            options (StateDictOptions): the options to control how
                model state_dict and optimizer state_dict should be loaded. See
                `StateDictOptions` for the details.

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys

        :type model_state_dict: typing.dict[str, ValueType]
        """
        from torch.distributed.fsdp._runtime_utils import _lazy_init
        for module in model.modules():
            if isinstance(module, FullyShardedDataParallel):
                _lazy_init(module, module)
        model_state_dict = _unflatten_model_state_dict(
            model, model_state_dict,
        )
        with gc_context():
            info = _verify_options(model, tuple(), optim_only=False, options=options)

            _verify_state_dict(model_state_dict, {}, info)
            return _load_model_state_dict(model, model_state_dict, info)

    def set_optimizer_state_dict(
        model: nn.Module,
        optimizers: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]],
        *,
        optim_state_dict,
        options = None,
    ) -> None:
        """Load the optimizers state_dict.

        The counterpart of ``get_optimizer_state_dict`` to set the state_dict to the
        optimizers. See ``set_state_dict`` for the detail usage.

        Args:
            model (nn.Module): the nn.Module to the model.
            optimizers (Union[Optimizer, Iterable[Optimizer]]):
                The optimizers that are used to optimize ``model``.
            optim_state_dict: OptimizerStateType:
                the optimizer state_dict to load.
            options (StateDictOptions): the options to control how
                model state_dict and optimizer state_dict should be loaded. See
                `StateDictOptions` for the details.

        Returns:
            None

        :type optim_state_dict: typing.OptimizerStateType
        """
        from torch.distributed.fsdp._runtime_utils import _lazy_init
        for module in model.modules():
            if isinstance(module, FullyShardedDataParallel):
                _lazy_init(module, module)
        with gc_context():
            optimizers = (
                (optimizers,)
                if isinstance(optimizers, torch.optim.Optimizer)
                else tuple(optimizers)
            )
            info = _verify_options(model, optimizers, optim_only=True, options=options)

            _verify_state_dict({}, optim_state_dict, info)
            _load_optim_state_dict(model, optimizers, optim_state_dict, info)


    # torch2.3 patch to fix https://github.com/pytorch/pytorch/issues/125740
    from torch.distributed.checkpoint.default_planner import (
        create_default_global_save_plan,
        DefaultSavePlanner,
        _validate_global_plan,
    )
    import dataclasses
    from collections import defaultdict, ChainMap

    from torch.distributed.checkpoint.planner import SavePlan, WriteItem
    from torch.distributed.checkpoint.metadata import MetadataIndex, Metadata

    def dedup_save_plans(all_plans: list[SavePlan]) -> list[SavePlan]:  # noqa: D103
        write_item_to_plan_indices: dict[MetadataIndex, set[int]] = defaultdict(set)
        write_item_idx_to_write_item: dict[MetadataIndex, WriteItem] = {}
        for plan_idx, plan in enumerate(all_plans):
            for write_item in plan.items:
                # map each write item to its plan
                write_item_to_plan_indices[write_item.index].add(plan_idx)
                write_item_idx_to_write_item[write_item.index] = write_item

        # put item in the plan with the smallest size and remove it from the other plan_indices
        to_remove: list[set] = [set() for _ in range(len(all_plans))]
        plan_to_size = [0] * len(all_plans)
        for write_item_idx, plan_indices in write_item_to_plan_indices.items():
            # this line is the fix, to keep the duplicated tensors on the same rank
            select_plan_idx = min(plan_indices, key=lambda plan_idx: plan_idx)

            write_item = write_item_idx_to_write_item[write_item_idx]
            # essentially ignores the storage size of anything that is not a tensor, since
            # we don't know how much storage they represent
            plan_to_size[select_plan_idx] += write_item.tensor_storage_size() or 1

            plan_indices.remove(select_plan_idx)
            for plan_idx in plan_indices:
                to_remove[plan_idx].add(write_item_idx)

        for plan_idx, remove_set in enumerate(to_remove):
            new_items = [
                write_item
                for write_item in all_plans[plan_idx].items
                if write_item.index not in remove_set
            ]
            all_plans[plan_idx] = dataclasses.replace(all_plans[plan_idx], items=new_items)

        return all_plans


    class SavePlannerWithDedupFix(DefaultSavePlanner):  # noqa: D101
        def create_global_plan(
            self, all_plans: list[SavePlan],
        ) -> tuple[list[SavePlan], Metadata]:
            all_plans = dedup_save_plans(all_plans)

            global_plan, metadata = create_default_global_save_plan(all_plans)

            if self.flatten_state_dict:
                # | does not work for Python 3.8 or older version.
                # merged_mappings = reduce(
                #     lambda x, y: x | y, (p.planner_data for p in global_plan)
                # )
                planner_data_dict = [p.planner_data for p in global_plan]
                merged_mappings = dict(ChainMap(*planner_data_dict))
                metadata = dataclasses.replace(metadata, planner_data=merged_mappings)

            if not _validate_global_plan(global_plan, metadata):
                raise ValueError('Failed to validate global plan')

            self.global_plan = global_plan
            self.metadata = metadata

            return self.global_plan, self.metadata

    from torch.utils._typing_utils import not_none
    from torch.distributed.device_mesh import DeviceMesh

    def create_child_mesh(
        self,
        device_mesh,
        mesh_dim_names: tuple[str],
    ):
        """Monkeypatch create_child_mesh to nightly version."""
        # swap the current dim to the last dim then reshape to flatten out other
        # dims, so we can just extract the list of ranks which contains cur_rank.
        mesh_dims = [
            not_none(device_mesh.mesh_dim_names).index(mesh_dim_name)
            for mesh_dim_name in mesh_dim_names
        ]
        cur_rank = device_mesh.get_rank()
        mesh = device_mesh.mesh
        all_mesh_dims = list(range(mesh.ndim))
        for mesh_dim in mesh_dims:
            # remove not pop b/c we want the value of the ind removed not it's position in the list
            # because this list dynamically changes.
            all_mesh_dims.remove(mesh_dim)

        mesh_sizes = [device_mesh.mesh.size(mesh_dim) for mesh_dim in mesh_dims]

        pg_ranks_by_dim = device_mesh.mesh.permute(
            *all_mesh_dims, *mesh_dims,
        ).reshape(-1, *mesh_sizes)

        for mesh_nd in pg_ranks_by_dim:
            if cur_rank in mesh_nd:
                sub_mesh = DeviceMesh(
                    device_mesh.device_type,
                    mesh_nd,
                    mesh_dim_names=mesh_dim_names,
                )
                res_sub_mesh = sub_mesh

        res_sub_mesh._dim_group_infos = [  # type: ignore
            device_mesh._dim_group_infos[mesh_dim] for mesh_dim in mesh_dims
        ]

        # Assign the current DeviceMesh as the parent of the child DeviceMesh.
        self.child_to_parent_mapping[res_sub_mesh] = device_mesh  # type: ignore
        return res_sub_mesh  # type: ignore

    from torch.distributed.device_mesh import _mesh_resources

    def device_mesh__init__(
        self,
        device_type: str,
        mesh,
        *,
        mesh_dim_names: Optional[tuple[str, ...]] = None,
    ) -> None:
        """Monkeypatch device mesh __init__ to nightly version."""
        self.device_type = device_type
        if isinstance(mesh, torch.Tensor) and mesh.device.type != 'cpu':
            raise ValueError(f'`mesh` must be a CPU tensor, got {mesh}')
        self.mesh = (
            mesh.detach().cpu()
            if isinstance(mesh, torch.Tensor)
            else torch.tensor(mesh, dtype=torch.int)
        )
        self.mesh_dim_names = mesh_dim_names

        # private field to pre-generate DeviceMesh's hash
        self._flatten_mesh_list = tuple(self.mesh.flatten().tolist())
        self._hash = hash((self._flatten_mesh_list, self.mesh.shape, id(self)))
        self._parent_mesh = _mesh_resources.get_parent_mesh(self)

        # Skip process group initialization if xla device.
        # TODO(yeounoh) implement DeviceMesh backend and register XLA backend.
        if device_type != 'xla':
            # always try to create default (world) pg, even if it is not initialized
            # already. The world pg is used for device mesh identity (rank) on each
            # process (we need to know if the current global rank is in the mesh or not).
            self._get_or_create_default_group()
            if not self._parent_mesh:
                self._init_process_groups()

    def device_mesh__getitem__(self, mesh_dim_names: Union[str, tuple[str]]) -> 'DeviceMesh':
        """Monkeypatch device_mesh __getitem__ to nightly version.

        Slice the current DeviceMesh based on the mesh_dim_name given to create a child
        DeviceMesh.

        Args:
            mesh_dim_name (str): the name of the mesh dimension of the parent DeviceMesh
            to create a child DeviceMesh for.

        Returns:
            A :class:`DeviceMesh` object

        The following program runs on each process/rank in an SPMD manner. In this example, we have 2
        hosts with 4 GPUs each.
        Calling mesh["tp"] on rank 0, 1, 2, 3 would return a 1D child DeviceMesh:([0, 1, 2, 3]).
        Calling mesh["tp"] on rank 4, 5, 6, 7 would return a 1D child DeviceMesh:([4, 5, 6, 7]).
        Calling mesh["dp"] on rank 0, 4 would return a 1D child DeviceMesh:([0, 4]).
        Calling mesh["dp"] on rank 1, 5 would return a 1D child DeviceMesh:([1, 5]).
        Calling mesh["dp"] on rank 2, 6 would return a 1D child DeviceMesh:([2, 6]).
        Calling mesh["dp"] on rank 3, 7 would return a 1D child DeviceMesh:([3, 7]).

        Example::
            >>> # xdoctest: +SKIP("no rank")
            >>> from torch.distributed.device_mesh import DeviceMesh
            >>>
            >>> # Initialize device mesh as (2, 4) to represent the topology
            >>> # of cross-host(dim 0), and within-host (dim 1).
            >>> mesh = DeviceMesh(device_type="cuda", mesh=[[0, 1, 2, 3],[4, 5, 6, 7]])
        """
        if not self.mesh_dim_names:
            raise RuntimeError('Cannot slice a DeviceMesh without mesh_dim_names.')

        mesh_dim_names = (
            (mesh_dim_names,) if isinstance(mesh_dim_names, str) else mesh_dim_names
        )

        error_msg = (
            f'Invalid mesh_dim_name {mesh_dim_names} specified. '
            f'Valid mesh_dim_names should be a contiguous subsequence of {self.mesh_dim_names}.'
        )

        # When the dimension slicing out is equal to the mesh dimensions of the current DeviceMesh,
        # we simply return self if the given slicing is valid.
        if mesh_dim_names == self.mesh_dim_names:
            return self
        # Check if the user-provided slicing is a valid contiguous subsequence of the mesh_dim_names
        # of the current DeviceMesh.
        elif len(mesh_dim_names) < len(self.mesh_dim_names):
            outermost_dim_name = mesh_dim_names[0]
            if outermost_dim_name not in self.mesh_dim_names:
                raise ValueError(error_msg)
            outermost_dim_idx = self.mesh_dim_names.index(outermost_dim_name)
            for i, j in zip(
                mesh_dim_names,
                self.mesh_dim_names[outermost_dim_idx : len(mesh_dim_names)],
            ):
                if i != j:
                    raise ValueError(error_msg)
        else:
            raise ValueError(error_msg)

        submesh = _mesh_resources.create_child_mesh(self, mesh_dim_names)
        return submesh
