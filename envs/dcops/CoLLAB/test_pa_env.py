from PersonalAssistant.make_benchmark import generate_instances
from PersonalAssistant.generate import make_prompts_vanilla
from PersonalAssistant.prompt_maker import (make_prompts_pa_DT_as_dict,
                                            make_prompts_pa_DPT_as_dict,
                                            make_prompts_pa_DPI_as_dict)




def main():
    cfg = dict(
        n_agents=15, max_degree=7,
        min_outfits_per_agent=5, max_outfits_per_agent=8,
        p_add_unary_color=0.7,
    )
    generate_instances(
        n_instances=30,
        cfg=cfg,
        dataset_dir="envs/datasets/collab_pa_DT_L",
        data_root="envs/PersonalAssistant/data",
        base_seed=00,
        make_collages=True,
        collage_cols=3,
        collage_thumb=(192, 192),
        make_prompts_fn=make_prompts_pa_DPI_as_dict,  # <— inject monolithic
        prompts_kwargs={"tone": "standard"},
        prompts_filename="monolithic.json",               # <— easy to find later
        progress=True,
    )


if __name__ == "__main__":
    main()