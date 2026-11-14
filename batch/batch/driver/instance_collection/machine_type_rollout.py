from typing import Optional


class MachineTypeRollout:
    """Moves a share of a pool's new autoscaled instances onto a machine type other than the pool's
    own, so the pool can adopt a new machine family gradually. A rolled-out instance's data disk is
    a network disk of the pool's data disk size, since a new machine family can't be assumed to
    offer the pool's local SSDs.

    This default rolls nothing out."""

    def machine_type(self, worker_type: str, cores: int) -> Optional[str]:  # pylint: disable=unused-argument
        """The machine type for one new autoscaled instance, or None for the pool's own."""
        return None
