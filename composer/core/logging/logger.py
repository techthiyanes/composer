# Copyright 2021 MosaicML. All Rights Reserved.

"""Base classes, functions, and variables for logger.

Attributes:

     LoggerData: Data value(s) to be logged. Can be any of the following types:
         ``str``; ``float``; ``int``; :class:`torch.Tensor`; ``Sequence[LoggerData]``;
         ``Mapping[str, LoggerData]``.
     LoggerDataDict: Name-value pair for data to be logged. Type ``Mapping[str, LoggerData]``.
         Example: ``{"accuracy", 21.3}``.
"""

from __future__ import annotations

import collections.abc
import operator
import time
from enum import IntEnum
from functools import reduce
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import coolname
import torch

from composer.utils import dist

if TYPE_CHECKING:
    from composer.core.logging.logger_destination import LoggerDestination
    from composer.core.state import State

__all__ = ["LoggerDestination", "Logger", "LogLevel", "LoggerData", "LoggerDataDict", "format_log_data_value"]

LoggerData = Union[str, float, int, torch.Tensor, List["LoggerData"], Dict[str, "LoggerData"]]
LoggerDataDict = Dict[str, LoggerData]


class LogLevel(IntEnum):
    """LogLevel denotes when in the training loop log messages are generated.

    Logging destinations use the LogLevel to determine whether to record a given
    metric or state change.

    Attributes:
        FIT: Logged once per training run.
        EPOCH: Logged once per epoch.
        BATCH: Logged once per batch.
    """
    FIT = 1
    EPOCH = 2
    BATCH = 3


class Logger:
    r"""Logger routes metrics to the :class:`.LoggerDestination`. Logger is what users call from within
    algorithms/callbacks. A logger routes the calls/data to any different number of destination
    :class:`.LoggerDestination`\\s (e.g., :class:`.FileLogger`, :class:`.InMemoryLogger`, etc.). Data to be logged should be
    of the type :attr:`~.logger.LoggerDataDict` (i.e., a ``{<name>: <value>}`` mapping).

    Args:
        state (State): The global :class:`~.core.state.State` object.
        destinations (Sequence[LoggerDestination]): A sequence of :class:`.LoggerDestination`\s to
            which logging calls will be sent.
        run_name (str, optional): The name for this training run.
            If not specified, a :doc:`coolname <coolname:/>` will be used like the following:

            .. testsetup:: composer.core.logging.logger.Logger.__init__.run_name

                import random
                import coolname

                coolname.replace_random(random.Random(0))

            .. doctest:: composer.core.logging.logger.Logger.__init__.run_name

                >>> str(time.time_ns()) + "-" + coolname.generate_slug(2)
                '1234-cool-name'

    Attributes:
        destinations (Sequence[LoggerDestination]):
            A sequence of :class:`~.LoggerDestination`\s to which logging calls will be sent.
    """

    def __init__(
            self,
            state: State,
            destinations: Sequence[LoggerDestination] = tuple(),
            run_name: Optional[str] = None,
    ):
        self.destinations = destinations
        if run_name is None:
            # prefixing with the time so experiments sorted alphabetically will
            # have the latest experiment last
            run_name = str(time.time_ns()) + "-" + coolname.generate_slug(2)
            run_name_list = [run_name]
            # ensure all ranks have the same experiment name
            dist.broadcast_object_list(run_name_list)
            run_name = run_name_list[0]
        self.run_name = run_name
        self._state = state

    def metric(self, log_level: Union[str, LogLevel], data: LoggerDataDict) -> None:
        """Log a metric to the :attr:`destinations`.

        Args:
            log_level (Union[str, LogLevel]): A :class:`LogLevel`.
            data (LoggerDataDict): The data to log.
        """
        if isinstance(log_level, str):
            log_level = LogLevel[log_level.upper()]

        for destination in self.destinations:
            destination.log_data(self._state.timer.get_timestamp(), log_level, data)

    def data_fit(self, data: LoggerDataDict) -> None:
        """Helper function for ``metric(LogLevel.FIT, data)``"""
        self.metric(LogLevel.FIT, data)

    def data_epoch(self, data: LoggerDataDict) -> None:
        """Helper function for ``self.metric(LogLevel.EPOCH, data)``"""
        self.metric(LogLevel.EPOCH, data)

    def data_batch(self, data: LoggerDataDict) -> None:
        """Helper function for ``self.metric(LogLevel.BATCH, data)``"""
        self.metric(LogLevel.BATCH, data)


def format_log_data_value(data: LoggerData) -> str:
    """Recursively formats a given log data value into a string.

    Args:
        data: Data to format.

    Returns:
        str: ``data`` as a string.
    """
    if data is None:
        return "None"
    if isinstance(data, str):
        return f"\"{data}\""
    if isinstance(data, int):
        return str(data)
    if isinstance(data, float):
        return f"{data:.4f}"
    if isinstance(data, torch.Tensor):
        if data.shape == tuple() or reduce(operator.mul, data.shape, 1) == 1:
            return format_log_data_value(data.cpu().item())
        return "Tensor of shape " + str(data.shape)
    if isinstance(data, collections.abc.Mapping):
        output = ['{ ']
        for k, v in data.items():
            assert isinstance(k, str)
            v = format_log_data_value(v)
            output.append(f"\"{k}\": {v}, ")
        output.append('}')
        return "".join(output)
    if isinstance(data, collections.abc.Iterable):
        return "[" + ", ".join(format_log_data_value(v) for v in data) + "]"
    raise NotImplementedError(f"Unable to format variable of type: {type(data)} with value {data}")
