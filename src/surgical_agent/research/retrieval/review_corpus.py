"""Small, attributable corpus copied from the existing review context.

These are project operational interpretations, NOT newly discovered official
annotation definitions. Every excerpt is already in the original reviewers'
input. The first comparison therefore tests retrieval/reorganization, not the
addition of independently verified medical knowledge. No frame labels, corpus
statistics, examples, model responses, or current-video memory are included.
"""

CORPUS_VERSION = "existing_review_boundaries_20260908_v1"

_BOUNDARIES = "scripts/run_semantic_candidate_trial.py::BOUNDARIES"
_INSTRUCTIONS = "scripts/run_semantic_candidate_trial.py::semantic_body.instructions"
_ONTOLOGY = "src/surgical_agent/perception/ontology_prompt.py::load_prompt_ontology_text"

# ``tasks`` attaches a definition to every legal label in that task. ``nodes``
# attaches only the specified ontology entities. The graph can traverse an IVT
# to its exact components before reaching a component-specific definition.
CORPUS = (
    {
        "id": "instrument_class",
        "text": "Instrument: some visible tool belongs to this class.",
        "tasks": ("instrument",),
        "nodes": (),
        "source_locator": _BOUNDARIES,
        "source_kind": "PROJECT_INTERPRETATION",
        "original_prompt_contains": True,
    },
    {
        "id": "verb_action",
        "text": "Verb: some tool performs this action on an object.",
        "tasks": ("verb",),
        "nodes": (),
        "source_locator": _BOUNDARIES,
        "source_kind": "PROJECT_INTERPRETATION",
        "original_prompt_contains": True,
    },
    {
        "id": "target_object",
        "text": (
            "Target: this anatomy is the object directly acted on by at least one "
            "instrument in the current frame. Background visibility, proximity, "
            "downstream organ deformation or transmitted force alone do not establish that Target."
        ),
        "tasks": ("target",),
        "nodes": (),
        "source_locator": _BOUNDARIES,
        "source_kind": "PROJECT_INTERPRETATION",
        "original_prompt_contains": True,
    },
    {
        "id": "ivt_relation",
        "text": (
            "IVT: this particular instrument-action-target relation, "
            "not three unrelated visible components."
        ),
        "tasks": ("ivt",),
        "nodes": (),
        "source_locator": _BOUNDARIES,
        "source_kind": "PROJECT_INTERPRETATION",
        "original_prompt_contains": True,
    },
    {
        "id": "grasp_retract_boundary",
        "text": (
            "Grasp and retract are distinct ontology labels: gripping tissue alone does "
            "not prove retract, and visible displacement alone does not identify the "
            "tissue at the tool tip."
        ),
        "tasks": (),
        "nodes": ("verb_0", "verb_1"),
        "source_locator": _BOUNDARIES,
        "source_kind": "PROJECT_INTERPRETATION",
        "original_prompt_contains": True,
    },
    {
        "id": "independent_component_support",
        "text": (
            "An IVT's rejection does not by itself refute its component labels: "
            "another tool/relation may support them."
        ),
        "tasks": ("ivt",),
        "nodes": (),
        "source_locator": _BOUNDARIES,
        "source_kind": "PROJECT_INTERPRETATION",
        "original_prompt_contains": True,
    },
    {
        "id": "frame_level_absence",
        "text": (
            "Deletion is a frame-level claim: any 1/2 requires WHOLE_FRAME scope after "
            "checking all tools. A negative observation of just one tool/region "
            "requires UNCLEAR and 3."
        ),
        "tasks": ("instrument", "verb", "target", "ivt"),
        "nodes": (),
        "source_locator": _INSTRUCTIONS,
        "source_kind": "PROJECT_INTERPRETATION",
        "original_prompt_contains": True,
    },
    {
        "id": "visible_null_relation",
        "text": "IVT 94-99 mean a visible instrument with null_verb and null_target.",
        "tasks": (),
        "nodes": tuple(f"ivt_{i}" for i in range(94, 100)) + ("verb_9", "target_14"),
        "source_locator": _ONTOLOGY,
        "source_kind": "ONTOLOGY",
        "original_prompt_contains": True,
    },
)
