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
    PUSH_BOX_CONTRACT_VERSION,
    PUSH_BOX_OBSERVATION_DIMS,
    HDMI_TASK_SPECS,
    HDMINetworkContract,
    HDMITaskSpec,
    HDMI_ACTION_JOINT_NAMES,
    HDMI_ACTION_SCALE,
    HDMI_DEFAULT_JOINT_POS,
    HDMI_PHYSICS_MATERIAL_COMBINE_MODE,
    HDMI_REFERENCE_JOINT_NAMES,
    REFERENCE_TO_ACTION_INDICES,
    FrozenHDMITeacherPolicy,
    HDMIJointPositionActionRuntime,
    HDMIObservationBatch,
    HDMIObservationHistory,
    PretrainedHDMIScaffold,
    get_hdmi_task_spec,
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
    "HDMINetworkContract",
    "HDMIObservationBatch",
    "HDMIObservationHistory",
    "HDMITaskSpec",
    "HDMI_TASK_SPECS",
    "HDMI_ACTION_JOINT_NAMES",
    "HDMI_ACTION_SCALE",
    "HDMI_DEFAULT_JOINT_POS",
    "HDMI_PHYSICS_MATERIAL_COMBINE_MODE",
    "HDMI_REFERENCE_JOINT_NAMES",
    "PRETRAINED_HDMI_CONTRACT_VERSION",
    "PUSH_BOX_CONTRACT_VERSION",
    "PUSH_BOX_OBSERVATION_DIMS",
    "PretrainedHDMIScaffold",
    "REFERENCE_TO_ACTION_INDICES",
    "ReferenceLibrary",
    "ReferenceMetadata",
    "ReferenceSource",
    "ScaffoldTask",
    "finite_difference",
    "get_hdmi_task_spec",
    "hdmi_mapping_to_reference",
    "load_hdmi_reference",
    "load_retargeted_omomo_reference",
    "omomo_retarget_mapping_to_reference",
    "reconstruct_contact_targets",
    "reference_action",
    "reference_to_action",
]
