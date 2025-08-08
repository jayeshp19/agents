from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

T = TypeVar("T")


def _parse_dict_field(value_type: type[Any], value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {k: parse_dataclass(value_type, v) for k, v in value.items()}
    if isinstance(value, list):
        key_name = None
        for candidate in ("id", "tool_id"):
            if any(f.name == candidate for f in fields(value_type)):
                key_name = candidate
                break
        if key_name is None:
            raise ValueError(f"Cannot determine key field for {value_type.__name__}")
        result: dict[str, Any] = {}
        for item in value:
            obj = parse_dataclass(value_type, item)
            key = getattr(obj, key_name)
            result[key] = obj
        return result
    raise ValueError("Dictionary field value must be a mapping or list")


def parse_dataclass(cls: type[T], data: Mapping[str, Any]) -> T:
    """Parse a mapping into the given dataclass type."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type")
    if not isinstance(data, Mapping):
        raise ValueError("Data must be a mapping")

    kwargs: dict[str, Any] = {}
    # Resolve forward references and postponed annotations
    try:
        module_globals = sys.modules[cls.__module__].__dict__
    except Exception:  # pragma: no cover - very unlikely
        module_globals = None
    resolved_hints = {}
    try:
        resolved_hints = get_type_hints(cls, globalns=module_globals)
    except Exception:
        # Fall back silently if type hints can't be resolved
        resolved_hints = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        field_type = resolved_hints.get(f.name, f.type)
        origin = get_origin(field_type)
        args = get_args(field_type)

        if value is None:
            kwargs[f.name] = None
            continue

        if is_dataclass(field_type):
            kwargs[f.name] = parse_dataclass(field_type, value)
        elif origin is list and args:
            inner = args[0]
            if is_dataclass(inner):
                kwargs[f.name] = [parse_dataclass(inner, v) for v in value]
            else:
                kwargs[f.name] = list(value)
        elif origin is dict and len(args) == 2 and is_dataclass(args[1]):
            kwargs[f.name] = _parse_dict_field(args[1], value)
        elif (origin is Union or origin is UnionType) and args:
            non_none_types = [a for a in args if a is not type(None)]
            parsed = False
            for candidate in non_none_types:
                cand_origin = get_origin(candidate)
                cand_args = get_args(candidate)
                # Direct dataclass in Union
                if is_dataclass(candidate):
                    kwargs[f.name] = parse_dataclass(candidate, value)
                    parsed = True
                    break
                # Union[list[Dataclass] | None]
                if cand_origin is list and cand_args:
                    inner = cand_args[0]
                    if is_dataclass(inner) and isinstance(value, list):
                        kwargs[f.name] = [parse_dataclass(inner, v) for v in value]
                        parsed = True
                        break
                # Union[dict[str, Dataclass] | list[Dataclass] | None]
                if cand_origin is dict and len(cand_args) == 2 and is_dataclass(cand_args[1]):
                    kwargs[f.name] = _parse_dict_field(cand_args[1], value)
                    parsed = True
                    break
            if not parsed:
                kwargs[f.name] = value
        else:
            kwargs[f.name] = value

    return cls(**kwargs)
