from .data_loader import TrackedData,TrackedData_infer,load_canonical_render_prams

def build_dataset(data_cfg, split,):

    return TrackedData(data_cfg, split)
