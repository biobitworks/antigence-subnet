"""
Per-instance registry for immune cell components.

ImmuneCellRegistry is instance-based (not a module-level global) so that
each miner can maintain its own cell configuration. This follows the
design decision D-04 to avoid global state.
"""

from __future__ import annotations

from antigence_subnet.miner.orchestrator.cells import ImmuneCellType


class ImmuneCellRegistry:
    """Registry that stores and retrieves immune cell instances by name.

    Instance-based: each miner creates its own registry. Validates that
    registered objects satisfy the ImmuneCellType Protocol at registration
    time.

    Usage:
        registry = ImmuneCellRegistry()
        registry.register("nk", NKCellStub())
        cell = registry.get("nk")
    """

    def __init__(self) -> None:
        self._cells: dict[str, ImmuneCellType] = {}

    def register(self, name: str, cell: ImmuneCellType) -> None:
        """Register an immune cell instance by name.

        Args:
            name: Unique identifier for the cell (e.g., 'nk', 'dc', 'bcell').
            cell: Object satisfying the ImmuneCellType Protocol.

        Raises:
            TypeError: If cell does not satisfy ImmuneCellType Protocol.
        """
        if not isinstance(cell, ImmuneCellType):
            raise TypeError(
                f"Cell '{name}' does not satisfy ImmuneCellType Protocol. "
                f"Expected process() method, got {type(cell).__name__}."
            )
        self._cells[name] = cell

    def get(self, name: str) -> ImmuneCellType | None:
        """Retrieve a registered cell by name, or None if not found.

        Args:
            name: Cell identifier to look up.

        Returns:
            The registered ImmuneCellType instance, or None.
        """
        return self._cells.get(name)

    def get_all(self) -> dict[str, ImmuneCellType]:
        """Return a copy of all registered cells.

        Returns:
            Dict mapping cell names to ImmuneCellType instances.
            Modifications to the returned dict do not affect the registry.
        """
        return dict(self._cells)
