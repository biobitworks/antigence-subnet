"""Tests for spec_version propagation from __init__.py to BaseNeuron (CORR-01).

Verifies that BaseNeuron.spec_version class attribute equals
antigence_subnet.__spec_version__ (1, not the fallback 0), so that
set_weights sends the correct version_key on chain.
"""

import antigence_subnet
from antigence_subnet.base.neuron import BaseNeuron


class TestSpecVersion:
    """Verify spec_version propagation."""

    def test_base_neuron_has_spec_version_attribute(self):
        """BaseNeuron class has a spec_version attribute (not relying on getattr fallback)."""
        assert hasattr(BaseNeuron, "spec_version"), (
            "BaseNeuron must have a spec_version class attribute"
        )

    def test_spec_version_equals_package_version(self):
        """BaseNeuron.spec_version equals antigence_subnet.__spec_version__."""
        assert BaseNeuron.spec_version == antigence_subnet.__spec_version__

    def test_spec_version_is_one_not_zero(self):
        """spec_version is 1, not the getattr fallback of 0."""
        assert BaseNeuron.spec_version == 1

    def test_instance_spec_version(self, mock_config):
        """An instantiated BaseNeuron also has spec_version == 1."""
        neuron = BaseNeuron(config=mock_config)
        assert neuron.spec_version == 1

    def test_getattr_fallback_not_needed(self):
        """getattr(BaseNeuron, 'spec_version', 0) returns 1, not 0."""
        # This is the exact pattern used in set_weights()
        assert getattr(BaseNeuron, "spec_version", 0) == 1
