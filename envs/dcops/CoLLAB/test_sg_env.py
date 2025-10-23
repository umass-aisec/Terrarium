from __future__ import annotations

# Use the updated writer that supports (make_prompts_fn, prompts_filename, include_prompts)
from SmartGrid.make_benchmark import write_instances_and_prompts as _write_min


from SmartGrid.prompt_maker import (make_prompts_powerlite_monolithic_parts_as_dict, 
                                        make_prompts_sg_DPI_as_dict, 
                                        make_prompts_sg_DPT_as_dict, 
                                        make_prompts_sg_DT_as_dict)



def main():
    """
    Dataset **with monolithic (by-parts) prompts**, PA-style.
    Stores prompts under 'monolithic.json' with sentinel keys:
      __TASK__, __DELIBERATION__, __JSON_MODE__
    index.json -> files.prompts = "monolithic.json"
    """
    cfg = dict(
        T=24,
        n_homes=8,
        tasks_per_home=(5,8),
        allowed_window_len=(2, 6),
        S_pattern="sin",
        S_base=12.0,
        S_amp=2.5,
    )
    dataset_dir = "envs/datasets/collab_smartgrid_DT_L"
    catalog_path = "envs/SmartGrid/data/devices.json"

    _write_min(
        n_instances=10,
        cfg=cfg,
        catalog_path=catalog_path,
        out_dir=dataset_dir,
        base_seed=0,  # 77..126
        include_prompts=True,
        make_prompts_fn=make_prompts_sg_DT_as_dict,  # <- by-parts maker
        prompts_filename="monolithic.json",  # <- matches PA
        include_viz=True,
    )
    print(f"[done] Wrote monolithic prompts dataset at {dataset_dir}")


if __name__ == "__main__":
    main()