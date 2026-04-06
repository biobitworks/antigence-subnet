"""Tests for ImmuneCellType Protocol, cell stubs, and ImmuneCellRegistry.

Covers: Protocol runtime checking, stub conformance, registry CRUD, type validation.
"""

import numpy as np
import pytest

from antigence_subnet.miner.detector import DetectionResult
from antigence_subnet.miner.orchestrator.cells import (
    BCellStub,
    DendriticCellStub,
    ImmuneCellType,
    NKCellStub,
)
from antigence_subnet.miner.orchestrator.registry import ImmuneCellRegistry


class TestImmuneCellTypeProtocol:
    """Tests for ImmuneCellType Protocol runtime checking."""

    def test_protocol_is_runtime_checkable(self):
        """ImmuneCellType supports isinstance() checks."""

        class Conforming:
            def process(
                self,
                features: np.ndarray,
                prompt: str,
                output: str,
                code: str | None = None,
                context: str | None = None,
            ) -> DetectionResult | None:
                return None

        obj = Conforming()
        assert isinstance(obj, ImmuneCellType)

    def test_non_conforming_class_fails_isinstance(self):
        """A class without process() does NOT satisfy ImmuneCellType."""

        class NotConforming:
            def detect(self, x):
                return x

        obj = NotConforming()
        assert not isinstance(obj, ImmuneCellType)

    def test_empty_class_fails_isinstance(self):
        """A class with no methods does not satisfy ImmuneCellType."""

        class Empty:
            pass

        assert not isinstance(Empty(), ImmuneCellType)


class TestCellStubs:
    """Tests for NK, DendriticCell, and BCell stub classes."""

    def test_nk_stub_satisfies_protocol(self):
        """NKCellStub satisfies isinstance(obj, ImmuneCellType)."""
        assert isinstance(NKCellStub(), ImmuneCellType)

    def test_dendritic_stub_satisfies_protocol(self):
        """DendriticCellStub satisfies isinstance(obj, ImmuneCellType)."""
        assert isinstance(DendriticCellStub(), ImmuneCellType)

    def test_bcell_stub_satisfies_protocol(self):
        """BCellStub satisfies isinstance(obj, ImmuneCellType)."""
        assert isinstance(BCellStub(), ImmuneCellType)

    def test_nk_stub_process_returns_none(self):
        """NKCellStub.process() returns None (stub behavior)."""
        cell = NKCellStub()
        result = cell.process(np.zeros(10), "prompt", "output")
        assert result is None

    def test_dendritic_stub_process_returns_none(self):
        """DendriticCellStub.process() returns None (stub behavior)."""
        cell = DendriticCellStub()
        result = cell.process(np.zeros(10), "prompt", "output")
        assert result is None

    def test_bcell_stub_process_returns_none(self):
        """BCellStub.process() returns None (stub behavior)."""
        cell = BCellStub()
        result = cell.process(np.zeros(10), "prompt", "output")
        assert result is None


class TestImmuneCellRegistry:
    """Tests for ImmuneCellRegistry CRUD and type validation."""

    def test_registry_starts_empty(self):
        """ImmuneCellRegistry() starts with empty cells dict."""
        registry = ImmuneCellRegistry()
        assert registry.get_all() == {}

    def test_registry_register_and_get(self):
        """registry.register('nk', cell) stores cell; registry.get('nk') returns it."""
        registry = ImmuneCellRegistry()
        nk = NKCellStub()
        registry.register("nk", nk)
        assert registry.get("nk") is nk

    def test_registry_get_nonexistent(self):
        """registry.get('nonexistent') returns None."""
        registry = ImmuneCellRegistry()
        assert registry.get("nonexistent") is None

    def test_registry_get_all(self):
        """registry.get_all() returns dict of all registered cells."""
        registry = ImmuneCellRegistry()
        nk = NKCellStub()
        dc = DendriticCellStub()
        registry.register("nk", nk)
        registry.register("dc", dc)
        all_cells = registry.get_all()
        assert len(all_cells) == 2
        assert all_cells["nk"] is nk
        assert all_cells["dc"] is dc

    def test_registry_get_all_returns_copy(self):
        """registry.get_all() returns a copy, not the internal dict."""
        registry = ImmuneCellRegistry()
        nk = NKCellStub()
        registry.register("nk", nk)
        all_cells = registry.get_all()
        all_cells["injected"] = NKCellStub()
        assert registry.get("injected") is None

    def test_registry_rejects_non_protocol(self):
        """registry.register() with non-protocol object raises TypeError."""

        class NotACell:
            pass

        registry = ImmuneCellRegistry()
        with pytest.raises(TypeError):
            registry.register("bad", NotACell())
