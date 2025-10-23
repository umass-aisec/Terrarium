from MeetingScheduling.make_benchmark import generate_instances
#from MeetingScheduling.prompt_maker import make_prompts_vanilla

# If you placed the monolithic prompter adapter in a different module, adjust this import accordingly.
# Example assuming you saved the earlier class+adapter in MeetingScheduling.mono_prompter:
from MeetingScheduling.prompt_maker import make_prompts_monolithic_parts_as_dict, make_prompts_ms_DT_as_dict


def main():
    # Dataset + MONOLITHIC PARTS prompts (task/deliberation/json-mode under sentinel keys)
    cfg = dict(
        n_agents=10,
        n_meetings=15,
        data_root="MeetingAssistant/data",
        max_attendees_per_meeting=4,
        p_zoom=0.30,
        min_prefs_per_agent=3,
        max_prefs_per_agent=6,
    )
    generate_instances(
        n_instances=10,
        cfg=cfg,
        dataset_dir="envs/datasets/collab_meeting_DT_L",
        base_seed=0,  # seeds instances: 99..103
        make_prompts_fn=make_prompts_ms_DT_as_dict,   # <= monolithic, 3-part prompts
        prompts_kwargs={"tone": "standard"},                     # viz_map is passed automatically by generator if available
        prompts_filename="prompts_parts.json",                   # contains __TASK__, __DELIBERATION__, __JSON_MODE__
    )


if __name__ == "__main__":
    main()
