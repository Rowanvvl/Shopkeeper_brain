import os
import re
import sys
from pathlib import Path

from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.task_utils import add_done_task, add_running_task

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    md_path = state.get('md_path')
    if not md_path:
        logger.error("路径不存在")
        raise ValueError("路径不存在")
    md_path_obj = Path(md_path)
    if not md_path_obj.exists():
        logger.error("文件不存在")
        raise ValueError("文件不存在")

    md_content = md_path_obj.read_text(encoding='utf-8')
    state['md_content'] = md_content

    md_images_dir_obj = md_path_obj.parent / "images"
    return md_content,md_path_obj,md_images_dir_obj

@step_log("step_2_scan_images")
def step_2_scan_images(md_content, md_images_dir_obj):
    image_info_list = []

    for image_file_obj in md_images_dir_obj.iterdir():
        image_name = image_file_obj.name
        image_path = str(image_file_obj)
        if not is_supported_image(image_name):
            logger.warning("不支持该图片格式")
            continue
        pattern = re.compile(r"!\[.*?\]\(.*?"+re.escape(image_name)+r".*?\)")

        search_match = pattern.search(md_content)
        if not search_match:
            logger.warning(f"{image_name}在md_content中没有被引用")
            continue
        start = search_match.start()
        end = search_match.end()

        pre_content = md_content[max(0,start-100):start]
        post_content = md_content[end:min(end+100,len(md_content))]

        image_info_list.append((image_name,image_path,(pre_content,post_content)))

    return image_info_list

@step_log("step_3_image_summary")
def step_3_image_summary(image_info_list, stem):
    pass


@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    """
    add_running_task(state['task_id'], 'node_md_img')
    # 1. 获取和校验参数
    md_content,md_path_obj, md_images_dir_obj  = step_1_validate_and_get_data(state)
    if (not md_images_dir_obj.is_dir()) or len(list(md_images_dir_obj.rglob("*.jpg"))) == 0:
        logger.info("该文档没有图片内容，跳出该节点")
        return state
    # 2. 获取图片的信息和上下文
    # list((图片名,图片地址,(前缀,后缀)))  =  scan_images(md_content,md_images_dir_obj)
    image_info_list: list[tuple[str, str, tuple[str, str]]] = step_2_scan_images(md_content, md_images_dir_obj)
    image_summary_dict: dict[str, str] = step_3_image_summary(image_info_list, md_path_obj.stem)
    add_done_task(state['task_id'], 'node_md_img')
    return state

if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(r"D:\Develop\Shopkeeper_brain\output\hak180产品安全手册\hak180产品安全手册.md")

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")