from __future__ import annotations

from typing import Any, Callable


class Registry:
    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, Callable[..., Any]] = {}

    def register(self, name: str | None = None) -> Callable:
        def decorator(fn: Callable) -> Callable:
            key = name or fn.__name__
            if key in self._entries:
                raise ValueError(f"{self._name}: '{key}' is already registered")
            self._entries[key] = fn
            return fn
        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._entries:
            available = ", ".join(sorted(self._entries))
            raise KeyError(f"{self._name}: '{name}' not found. Available: {available}")
        return self._entries[name]

    def build(self, name: str, **kwargs: Any) -> Any:
        return self.get(name)(**kwargs)

    def list_available(self) -> list[str]:
        return sorted(self._entries.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)


OPTIMIZER_REGISTRY: Registry = Registry("optimizer")
TASK_REGISTRY: Registry = Registry("task")
