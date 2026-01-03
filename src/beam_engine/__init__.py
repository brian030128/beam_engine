__version__ = "0.1.0"

# Lazy imports to avoid circular import issues
def __getattr__(name):
    if name in ["BeamSearchGenerator", "run_huggingface_beam_search", "demo_diverse_beam_search"]:
        from .beam import *
        return globals()[name]
    elif name in ["PageTable"]:
        from .page_table import *
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")