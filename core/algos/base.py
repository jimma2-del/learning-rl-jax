import jax.numpy as jnp
import jax

from jax.typing import ArrayLike
from chex import dataclass
from typing import TypeVar, Generic, Protocol, TypeAlias

TScheduleValue = TypeVar('TScheduleValue')

class Schedule(Generic[TScheduleValue], Protocol):
    def __call__(self, steps: int) -> TScheduleValue:
        ...

Scheduleable: TypeAlias = TScheduleValue | Schedule[TScheduleValue]

def resolve_scheduleable(scheduleable: Scheduleable[TScheduleValue], steps: int) -> TScheduleValue:
    if callable(scheduleable): return scheduleable(steps)
    return scheduleable
