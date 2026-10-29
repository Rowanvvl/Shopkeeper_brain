import base64
import os
import re
import sys
from mimetypes import guess_type
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject

from common.config.lm_config import lm_config
from common.config.minio_config import minio_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.clients.minio_utils import get_minio_client
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.rate_limit_utils import apply_api_rate_limit
from utils.task_utils import add_done_task, add_running_task
from dotenv import load_dotenv
load_dotenv()
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
def step_3_image_summary(image_info_list, root_folder):
    summary_image_dict = {}
    vl_model = get_llm_client(model=lm_config.vl_model)
    for image_name,image_path,image_content in image_info_list:
        image_prompt_text = load_prompt("image_summary",root_folder=root_folder,image_content=image_content)
        image_path_obj = Path(image_path)
        image_base64_str = base64.b64encode(image_path_obj.read_bytes()).decode(encoding="utf-8")
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{guess_type(image_name)[0]};base64,{image_base64_str}"}
                },  # 图片
                {
                    "type": "text", "text": image_prompt_text
                }  # 文本
            ]
        )
        chains = vl_model | StrOutputParser()
        apply_api_rate_limit()
        image_summary=chains.invoke([message])
        summary_image_dict[image_name] = image_summary
    return summary_image_dict


@step_log("step_4_upload_images_get_url")
def step_4_upload_images_get_url(image_info_list, stem):
    minio_client = get_minio_client()
    list_object = minio_client.list_objects(
        bucket_name=minio_config.bucket_name,
        prefix=minio_config.minio_img_dir[1:]+"/"+stem,
        recursive=True
    )
    delete_object_list = [DeleteObject(obj.object_name) for obj in list_object]
    errors = minio_client.remove_objects(
        bucket_name=minio_config.bucket_name,
        delete_object_list=delete_object_list
    )
    for error in errors:
        logger.warning("删除图片出现问题")

    image_url_dict = {}
    for image_name,image_path,_ in image_info_list:
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=minio_config.minio_img_dir+"/"+stem+"/"+image_name,
                file_path=image_path,
                content_type=guess_type(image_name)[0]
            )
            url = f"http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_name}"
            image_url_dict[image_name] = url
            logger.debug(f"{image_name}已经完成上传,对应的地址为:{url}")
        except Exception as e:
            logger.warning(f"{image_name}上传失败，跳过")
            continue

    return image_url_dict

@step_log("step_5_md_content_image_replace")
def step_5_md_content_image_replace(md_content, summary_image_dict, image_url_dict):
    for image_name,image_summary in summary_image_dict.items():
        image_url = image_url_dict.get(image_name)
        reg = re.compile(r"\!\[.*?\]\(.*?"+re.escape(image_name)+r".*?\)")
        md_content = reg.sub(lambda _:f"![{image_summary}]({image_url})",md_content)
        logger.debug(f"已经完成:{image_name}图片的替换,替换入的描述:{image_summary},替换的地址:{image_url}")
    return md_content

@step_log("step_6_md_content_image_replace")
def step_6_backup_new_md_content(md_content_new, md_path_obj):
    md_path_obj_new = md_path_obj.with_name(f"{md_path_obj.stem}_new.md")
    md_path_obj_new.write_text(data=md_content_new,encoding="utf-8")
    logger.info(f"已经将新内容备份到{str(md_path_obj_new)}")
    return md_path_obj_new


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
    # 3. 调用视觉模型 image_info_list [图片名,图片地址,上下文] -> {图片名:总结内容}
    image_summary_dict: dict[str, str] = step_3_image_summary(image_info_list, md_path_obj.stem)
    # 4. 将图片信息传递到minio的服务器,并获取图片和对应的网络地址
    image_url_dict: dict[str, str] = step_4_upload_images_get_url(image_info_list, md_path_obj.stem)
    # 5. 原md_content内图片 ![](./xx) -> ![图片的描述](图片的网络地址)
    # summary_image_dict -> image_name : 图片的描述  image_url_dict  image_name : url
    md_content_new: str = step_5_md_content_image_replace(md_content, image_summary_dict, image_url_dict)
    # 6. 将md_content_new进行磁盘化备份(1.新和旧进行对比测试 2.后续节点md_content内容丢失->再次读取)
    md_path_obj_new: Path = step_6_backup_new_md_content(md_content_new, md_path_obj)
    # 8. 更新state
    state['md_content'] = md_content_new
    state['md_path'] = str(md_path_obj_new)
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