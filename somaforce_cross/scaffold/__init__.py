"""Standalone pretrained HDMI runtime and canonical reference interfaces."""

from somaforce_cross.scaffold.contracts import (
    G1_BODY_JOINT_NAMES,
    G1_FULL_JOINT_NAMES,
    ScaffoldTask,
)
from somaforce_cross.scaffold.hdmi_adapter import (
    hdmi_mapping_to_reference,
    load_hdmi_reference,
)
from somaforce_cross.scaffold.omomo_adapter import (
    load_retargeted_omomo_reference,
    omomo_retarget_mapping_to_reference,
)
from somaforce_cross.scaffold.pretrained_hdmi import (
    CONTRACT_VERSION as PRETRAINED_HDMI_CONTRACT_VERSION,
    HDMI_ACTION_JOINT_NAMES,
    HDMI_ACTION_SCALE,
    HDMI_REFERENCE_JOINT_NAMES,
    REFERENCE_TO_ACTION_INDICES,
    FrozenHDMITeacherPolicy,
    HDMIJointPositionActionRuntime,
    HDMIObservationBatch,
    HDMIObservationHistory,
    PretrainedHDMIScaffold,
    reference_action,
    reference_to_action,
)
from somaforce_cross.scaffold.reference_library import ReferenceLibrary
from somaforce_cross.scaffold.reference_schema import (
    CanonicalReferenceEpisode,
    ReferenceMetadata,
    ReferenceSource,
    finite_difference,
    reconstruct_contact_targets,
)

__all__ = [
    "CanonicalReferenceEpisode",
    "FrozenHDMITeacherPolicy",
    "G1_BODY_JOINT_NAMES",
    "G1_FULL_JOINT_NAMES",
    "HDMIJointPositionActionRuntime",
    "HDMIObservationBatch",
    "HDMIObservationHistory",
    "HDMI_ACTION_JOINT_NAMES",
    "HDMI_ACTION_SCALE",
    "HDMI_REFERENCE_JOINT_NAMES",
    "PRETRAINED_HDMI_CONTRACT_VERSION",
    "PretrainedHDMIScaffold",
    "REFERENCE_TO_ACTION_INDICES",
    "ReferenceLibrary",
    "ReferenceMetadata",
    "ReferenceSource",
    "ScaffoldTask",
    "finite_difference",
    "hdmi_mapping_to_reference",
    "load_hdmi_reference",
    "load_retargeted_omomo_reference",
    "omomo_retarget_mapping_to_reference",
    "reconstruct_contact_targets",
    "reference_action",
    "reference_to_action",
]
