"""Scoped training numeric checks; ordinary callers validate immediately."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from threading import local
from typing import cast

import torch
from torch import Tensor


class _CheckScope(local):
    def __init__(self) -> None:
        self.active = False
        self.collector: _TrainingNumericChecks | None = None


_scope = _CheckScope()


def _collecting_training_checks() -> bool:
    return _scope.active


def _check_numeric_many(
    checks: Sequence[tuple[Tensor, str]], *, _compiled_async: bool = False
) -> None:
    if not checks:
        return
    for predicate, _message in checks:
        if predicate.dtype != torch.bool or predicate.ndim != 0:
            raise ValueError("numeric checks require scalar bool predicates")
    if torch.compiler.is_compiling():
        # Dynamo guards this thread-local bool. Public calls keep their original
        # eager error boundary; the explicit training scope captures GPU checks.
        if _scope.active or _compiled_async:
            for predicate, message in checks:
                torch._assert_async(predicate, message)
            return
    _eager_numeric_checks(checks)


_NumericValidation = Callable[[Sequence[tuple[Tensor, str]]], None]
_disable_checks = cast(Callable[[_NumericValidation], _NumericValidation], torch.compiler.disable)


@_disable_checks
def _eager_numeric_checks(checks: Sequence[tuple[Tensor, str]]) -> None:
    collector = _scope.collector
    if collector is not None:
        if collector.failure is None:
            collector.pending.extend((predicate.detach(), message) for predicate, message in checks)
        return
    _evaluate(checks)


def _evaluate(checks: Sequence[tuple[Tensor, str]]) -> None:
    by_device: dict[torch.device, list[Tensor]] = {}
    for predicate, _message in checks:
        by_device.setdefault(predicate.device, []).append(predicate)
    if not all(bool(torch.stack(predicates).all().item()) for predicates in by_device.values()):
        for predicate, message in checks:
            if not bool(predicate.item()):
                raise ValueError(message)


def _check_numeric(predicate: Tensor, message: str, *, _compiled_async: bool = False) -> None:
    _check_numeric_many(((predicate, message),), _compiled_async=_compiled_async)


class _TrainingNumericChecks:
    """Own computed local predicates until the trainer's update boundary."""

    def __init__(self) -> None:
        self.pending: list[tuple[Tensor, str]] = []
        self.failure: Exception | None = None

    @contextmanager
    def collect(self, *, enabled: bool) -> Iterator[None]:
        if not enabled:
            yield
            return
        if _scope.active:
            raise RuntimeError("training numeric-check scopes cannot nest")
        _scope.active, _scope.collector = True, self
        try:
            yield
        except Exception as error:
            # This path is local only. The trainer owns all rank coordination.
            raise self.latch_failure(error)
        finally:
            _scope.active, _scope.collector = False, None

    def latch_failure(self, error: Exception) -> Exception:
        if self.failure is None:
            try:
                self._check_immediately()
            except Exception as earlier:
                error = earlier
            self.failure = error
        return self.failure

    def _check_immediately(self) -> None:
        _evaluate(self.pending)

    def check(self) -> None:
        if self.failure is not None:
            raise self.failure
        _evaluate(self.pending)
        self.pending.clear()
