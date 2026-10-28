import json
import sys
from pathlib import Path

from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.task_utils import add_running_task, add_done_task

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    chunks = state.get("chunks")
    if not chunks:
        md_path = state.get("md_path")
        md_path_obj = Path(md_path)
        if md_path_obj.is_file():
            json_path_obj = md_path_obj.with_name(f"{md_path_obj.stem}.json")
            if json_path_obj.is_file():
                chunks = json.load(json_path_obj.read_text(encoding="utf-8"))
                for chunk in chunks:
                    chunk["item_name"] = state.get("item_name")
                state["chunks"] = chunks
            else:
                logger.error("chunks没值")
                raise ValueError("chunks没值，备份文件不存在")
        else:
            logger.error("chunks没值")
            raise ValueError("文件不存在")
    return chunks


@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    """
    add_running_task(state["task_id"],"node_bge_embedding")
    # 1. 获取并校验参数 [如果chunks ->备份文件读取 -> item_name]
    chunks = step_1_validate_and_get_data(state)

    add_done_task(state["task_id"],"node_bge_embedding")
    return state