from .utils import get_model_config, post_process
import json
import jinja2
from .api_client import LLM
from .prompts.utils import ADJUST_JSON_FORMAT_PROMPT


async def llm_inference_json(params, llm_service, retry_times=3, check_times=3):
    repair_config = get_model_config("json_repair_model")
    llm_check_service = LLM(
        model=repair_config["model"],
        system_prompt=repair_config["system_prompt"],
        temperature=repair_config["temperature"],
        top_p=repair_config["top_p"],
        max_new_tokens=repair_config["max_new_tokens"],
        template=jinja2.Template(ADJUST_JSON_FORMAT_PROMPT),
    )
    status = False
    for retry_id in range(retry_times):
        ret = (await llm_service.request(params)).response
        llm_service.last_output = ret
        for check_id in range(check_times):
            try:
                processed_ret = post_process(ret)
                ret_json = json.loads(processed_ret)
                if not isinstance(ret_json, list) and (not isinstance(ret_json, dict)):
                    raise ValueError(f"Expected a JSON object or array: {ret_json}")
                status = True
                break
            except Exception:
                ret = (await llm_check_service.request({"Ret": ret})).response
                llm_service.last_output = ret
        if status:
            break
    if not status:
        return {}
    return ret_json
