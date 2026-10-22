from . import scene_review as review

EVIDENCE_FIELDS = ("plot_summary", "key_actions", "conflicts_and_tension")


def evidence_catalog(scenes):
    catalog = {}
    for scene in scenes:
        for field in EVIDENCE_FIELDS:
            for text in review.text_values(scene.get(field, [])):
                if text.strip():
                    catalog[f"S{len(catalog) + 1:04d}"] = {
                        "scene_id": scene["scene_id"],
                        "field": field,
                        "text": text,
                    }
    if not catalog:
        raise ValueError("No scene action evidence available")
    return catalog
