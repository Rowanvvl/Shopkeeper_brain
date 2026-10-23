import sys
from pathlib import Path

from common.logging.logger import logger, node_log
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.task_utils import add_running_task, add_done_task


@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    add_running_task(state["task_id"], "node_entry")

    local_file_path = state.get("local_file_path")

    if not local_file_path:
        logger.warning("文件路径为空，直接终止当前导入流程")
        return state

    if local_file_path.endswith(".md"):
        state["md_path"] = local_file_path
        state["pdf_path"] = None
        state["is_pdf_read_enabled"] = False
        state["is_md_read_enabled"] = True

    elif local_file_path.endswith(".pdf"):
        state["md_path"] = None
        state["pdf_path"] = local_file_path
        state["is_pdf_read_enabled"] = True
        state["is_md_read_enabled"] = False

    else:
        logger.warning(
            f'不支持{local_file_path}该文件类型,终止流程'
        )
        return state

    state["file_title"] = Path(local_file_path).stem

    add_done_task(state["task_id"], "node_entry")

    return state

if __name__ == '__main__':

    # 单元测试：覆盖不支持类型、MD、PDF三种场景
    logger.info("===== 开始node_entry节点单元测试 =====")

    # 测试1: 不支持的TXT文件
    test_state1 = create_default_state(
        task_id="test_task_001",
        local_file_path="联想海豚用户手册.txt"
    )
    # node_entry(test_state1)

    # 测试2: MD文件
    test_state2 = create_default_state(
        task_id="test_task_002",
        local_file_path="小米用户手册.md"
    )
    # node_entry(test_state2)

    # 测试3: PDF文件
    test_state3 = create_default_state(
        task_id="test_task_003",
        local_file_path="万用表的使用.pdf"
    )
    node_entry(test_state3)

    logger.info("===== 结束node_entry节点单元测试 =====")