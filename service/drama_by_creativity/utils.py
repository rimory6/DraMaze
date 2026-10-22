import os
import yaml


def require_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value


def post_process(ret):
    ret = ret.replace("```json\n", "").replace("\n```", "").strip()
    return ret


def read_yaml(data_path):
    with open(data_path, "r", encoding="utf-8") as file:
        configs = yaml.safe_load(file)
    return configs


MODEL_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model_configs.yaml"
)


def calc_scene_nums_by_per_scene_word(drama_config, min_word_count, max_word_count):
    import math

    per_scene_config = drama_config.get("per_scene_word_range")
    if not per_scene_config or not min_word_count or (not max_word_count):
        return (
            drama_config.get("scene_nums_min", 1),
            drama_config.get("scene_nums_max", 10),
        )
    per_scene_min = per_scene_config["min"]
    per_scene_max = per_scene_config["max"]
    scene_nums_min = max(1, math.ceil(min_word_count / per_scene_max))
    scene_nums_max = max(
        scene_nums_min,
        drama_config.get("scene_nums_max", 0),
        math.ceil(max_word_count / per_scene_min),
    )
    return (scene_nums_min, scene_nums_max)


def calc_word_nums_by_word_count(drama_config, min_word_count, max_word_count):
    if (
        min_word_count
        and max_word_count
        and (min_word_count > 0)
        and (max_word_count > 0)
    ):
        return f"{int(min_word_count)}-{int(max_word_count)}"
    elif min_word_count and min_word_count > 0:
        return f"{int(min_word_count)}-{int(min_word_count * 1.4)}"
    elif max_word_count and max_word_count > 0:
        return f"{int(max_word_count * 0.7)}-{int(max_word_count)}"
    else:
        return drama_config.get("word_nums", "300")


def get_model_config(model_type):
    configs = read_yaml(MODEL_CONFIG_PATH)
    return configs[model_type]

