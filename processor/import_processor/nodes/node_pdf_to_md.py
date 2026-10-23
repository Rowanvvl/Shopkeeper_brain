import os
import shutil
import sys
import time
from pathlib import Path

import requests
from requests import session

from common.config.mineru_config import mineru_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.path_util import PROJECT_ROOT
from utils.task_utils import add_done_task, add_running_task

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir")

    if not pdf_path:
        logger.error("No pdf path provided")
        raise ValueError("No pdf path provided")

    if not local_dir:
        logger.warning("No local path provided")
        local_dir = PROJECT_ROOT / "output"
        state["local_dir"] = str(local_dir)

    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    if not pdf_path_obj.is_file():
        logger.error("Pdf path does not exist")
        raise ValueError("Pdf path does not exist")
    if not local_dir_obj.is_dir():
        logger.warning("Local path does not exist")
        local_dir_obj.mkdir(parents=True,exist_ok=True)
    return pdf_path_obj,local_dir_obj

@step_log("step_2_upload_and_poll")
def step_2_upload_and_poll(pdf_path_obj):
    if not mineru_config.base_url or not mineru_config.api_key:
        logger.error("No mineru base_url or api_key provided")
        raise ValueError("No mineru base_url or api_key provided")
    token = mineru_config.api_key
    url = "https://mineru.net/api/v4/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": "demo.pdf", "data_id": "abcd"}
        ],
        "model_version": "vlm"
    }
    response = requests.post(url,headers=header,json=data)
    http_state_code = response.status_code
    if http_state_code != 200:
        logger.error("HTTP status code not 200")
        raise RuntimeError("HTTP status code not 200")

    response_dict=response.json()
    response_dict_code = response_dict["code"]
    response_dict_msg = response_dict['msg']
    if response_dict_code != 0:
        logger.error(f"申请地址网络状态成功!但是业务失败!错误码:{response_dict_code},失败信息:{response_dict_msg}")
        raise RuntimeError(f"申请地址网络状态成功!但是业务失败!错误码:{response_dict_code},失败信息:{response_dict_msg}")
    batch_id = response_dict["data"]["batch_id"]
    urls = response_dict["data"]["file_urls"]

    if not urls:
        logger.error("No urls provided")
        raise ValueError("No urls provided")
    if not batch_id:
        logger.error("No batch_id provided")
        raise ValueError("No batch_id provided")
    file_upload_url = urls[0]
    logger.info(f"申请地址成功:{file_upload_url}")

    pdf_file_data = pdf_path_obj.read_bytes()
    with requests.Session() as session:
        session.trust_env = False
        upload_response = session.put(file_upload_url, data=pdf_file_data)
        if upload_response.status_code != 200:
            logger.error(f"上传文件失败,返回状态码为:{upload_response.status_code},请检查minerU配置!")
            raise RuntimeError(f"上传文件失败,返回状态码为:{upload_response.status_code},请检查minerU配置!")

    logger.info(f"向:{file_upload_url}上传{pdf_path_obj}成功")

    poll_url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    timeout = 600
    interval_time = 3
    start_time = time.time()
    while True:
        #判断是否超时
        if time.time() - start_time > timeout:
            logger.error("轮询超时，请检查配置")
            raise TimeoutError("轮询超时，请检查配置")
        #进行轮询
        try:
            poll_response = requests.get(poll_url, headers=header)
        except Exception as e:
            logger.warning("请求出现异常，请稍后重试!")
            time.sleep(interval_time)
            continue
        http_poll_status_code = poll_response.status_code
        if http_poll_status_code != 200:
            if 500 <= http_poll_status_code < 600:
                logger.warning(f"有可以修复的网络异常，状态码是{http_poll_status_code}")
                time.sleep(interval_time)
                continue
            else:
                logger.error(f"不可修复的网络状态异常{http_poll_status_code}")
                raise RuntimeError(f"不可修复的网络状态异常{http_poll_status_code}")
        poll_response_dict = poll_response.json()
        if poll_response_dict["code"] != 0:
            logger.error(f"轮询业务异常,错误码:{poll_response_dict['code']},失败信息:{poll_response_dict['msg']}")
            raise RuntimeError(f"轮询业务异常,错误码:{poll_response_dict['code']},失败信息:{poll_response_dict['msg']}")

        extract_result = poll_response_dict['data']['extract_result'][0]
        extract_result_state = extract_result["state"]

        if extract_result_state == "done":
            extract_result_url = extract_result["full_zip_url"]
            if not extract_result_url:
                logger.error(f"已经完成解析，但是地址为空")
                raise RuntimeError(f"已经完成解析，但是地址为空")
            logger.info(f">>>获取mineru服务器的解析结果：{extract_result_url}>>>")
            return extract_result_url
        elif extract_result_state == "failed":
            logger.error(f"已经完成了解析,但是失败了!!失败信息:{extract_result['err_msg']}")
            raise RuntimeError(f"已经完成了解析,但是失败了!!失败信息:{extract_result['err_msg']}")
        else:
            logger.warning(f"解析正在进行中,状态:{extract_result_state}!")
            time.sleep(interval_time)
            continue

@step_log("step_3_download_and_extract")
def step_3_download_and_extract(zip_url, local_dir_obj, stem):
    response = requests.get(zip_url, timeout=60)
    if response.status_code != 200:
        logger.error(f"从{zip_url}下载文件失败!")
        raise RuntimeError(f"从{zip_url}下载文件失败!")
    md_zip_path_obj = local_dir_obj / f"{stem}_op.zip"
    md_zip_path_obj.write_bytes(response.content)

    extract_path_obj = local_dir_obj / stem
    if extract_path_obj.is_dir():
        shutil.rmtree(extract_path_obj)
    extract_path_obj.mkdir(parents=True,exist_ok=True)
    shutil.unpack_archive(md_zip_path_obj,extract_path_obj)

    md_file_list = list(extract_path_obj.rglob("*.md"))
    if not md_file_list:
        logger.error(f"文件解压失败,在:{extract_path_obj}没有任何md文件!")
        raise RuntimeError(f"文件解压失败,在:{extract_path_obj}没有任何md文件!")

    target_md_obj = None
    for md_file_obj in md_file_list:
        if md_file_obj.name.lower() == "full.md":
            target_md_obj = md_file_obj
            logger.info(f"文件解压成功,但是名字是full.md,后续需要重命名")

    final_md_path_obj = target_md_obj.rename(target_md_obj.with_name(f"{stem}.md"))
    logger.info(f"文件{md_file_obj}解压成功")
    return final_md_path_obj



@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    # 1.注解日志和进行和完成任务状态处理
    add_running_task(state['task_id'], 'node_pdf_to_md')
    # 2. step_1_validate_paths 校验路径完整以及是否真实存在
    pdf_path_obj, local_dir_obj = step_1_validate_and_get_data(state)
    # 3. step_2_upload_and_poll
    zip_url = step_2_upload_and_poll(pdf_path_obj)
    # 4. step_3_download_and_extract
    md_path = step_3_download_and_extract(zip_url, local_dir_obj, pdf_path_obj.stem)
    # 5. 处理响应结果 state进行赋值处理
    state['md_path'] = md_path
    add_done_task(state['task_id'], 'node_pdf_to_md')
    return state

if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")