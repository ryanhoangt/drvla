
import re

COMPONENT_DIMS = {"paligemma": 2048, "action_expert": 1024}
NUM_LAYERS = 18

# The eight layers analysed in the paper.
PAPER_LAYERS = [
    "paligemma.layer_0.output",
    "paligemma.layer_5.output",
    "paligemma.layer_11.output",
    "paligemma.layer_17.output",
    "action_expert.layer_0.output",
    "action_expert.layer_5.output",
    "action_expert.layer_11.output",
    "action_expert.layer_17.output",
]

_LAYER_RE = re.compile(r"^(paligemma|action_expert)\.layer_(\d+)\.output$")


def parse_layer(name: str) -> tuple[str, int]:
    """Split a layer name into ``(component, index)``."""
    match = _LAYER_RE.match(name)
    if match is None:
        raise ValueError(
            f"Invalid layer name {name!r}. Expected 'paligemma.layer_<i>.output' "
            f"or 'action_expert.layer_<i>.output' with 0 <= i < {NUM_LAYERS}."
        )
    component, index = match.group(1), int(match.group(2))
    if not 0 <= index < NUM_LAYERS:
        raise ValueError(f"Layer index {index} in {name!r} is outside 0..{NUM_LAYERS - 1}.")
    return component, index


def layer_dim(name: str) -> int:
    """Residual-stream width of a layer."""
    component, _ = parse_layer(name)
    return COMPONENT_DIMS[component]


def short_name(name: str) -> str:
    """Compact label used in the paper, e.g. ``PG5`` or ``AE11``."""
    component, index = parse_layer(name)
    return f"{'PG' if component == 'paligemma' else 'AE'}{index}"


def get_layer_module(model, name: str):
    """Return the decoder block of an openpi ``PI0Pytorch`` model for a layer name."""
    component, index = parse_layer(name)
    experts = model.paligemma_with_expert
    if component == "paligemma":
        return experts.paligemma.language_model.layers[index]
    return experts.gemma_expert.model.layers[index]
