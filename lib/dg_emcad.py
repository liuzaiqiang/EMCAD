"""Public import entrypoint for the DG-EMCAD architecture.

The implementation is maintained in ``DG_EMCAD_reference`` to keep one
authoritative copy during the manual integration phase.  Training code should
always import the classes from this module.
"""

from DG_EMCAD_reference.lib.dg_emcad import (
    DGEMCADNet,
    DisagreementGuidedEMCAD,
    DisagreementGuidedMSCB,
)

__all__ = [
    "DGEMCADNet",
    "DisagreementGuidedEMCAD",
    "DisagreementGuidedMSCB",
]
