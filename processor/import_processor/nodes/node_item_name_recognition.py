import json
import sys
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import has_collection, DataType

from common.config.milvus_config import milvus_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.clients.milvus_utils import get_milvus_client
from utils.lm.embedding_utils import generate_embeddings
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.task_utils import add_running_task, add_done_task

# 主体识别上下文切片数：取前 K 个切片用于 LLM 识别
ITEM_NAME_CONTEXT_CHUNK_K = 5
# 主体识别上下文总字符数上限：防止上下文过长导致大模型输入超限
ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS = 2000

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    chunks = state["chunks"]
    file_title = state["file_title"]
    md_path = state["md_path"]
    md_path_obj = Path(md_path)

    if not chunks:
        if md_path_obj.is_file():
            json_path_obj = Path(md_path).with_name(Path(md_path).stem+".json")
            if json_path_obj.is_file():
                chunks = json.loads(json_path_obj.read_text(encoding="utf-8"))
                state["chunks"] = chunks
                logger.warning("从备份数据中读取")
            else:
                logger.error("没chunks,备份数据为空")
                raise FileNotFoundError("没chunks,备份数据为空")
        else:
            logger.error(f"{md_path_obj} is not a directory")
            raise ValueError("没chunks,路径中也没有文件")
    if not file_title:
        file_title = md_path_obj.stem
        state["file_title"] = file_title
        logger.warning("赋名")

    return chunks, file_title

@step_log("step_2_call_llm_return_item_name")
def step_2_call_llm_return_item_name(chunks, file_title):
    llm_model = get_llm_client()

    context = ""
    for chunk in chunks[:ITEM_NAME_CONTEXT_CHUNK_K]:
        context += f"标题{chunk.get("parent_title")}内容{chunk.get("content")}"
    context = context[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]
    item_name_reco_prompt_text = load_prompt("item_name_recognition",file_title = file_title,context = context)
    msg = HumanMessage(
        content=item_name_reco_prompt_text,
    )
    chains = llm_model | StrOutputParser()
    item_name = chains.invoke([msg])

    if not item_name:
        item_name = file_title
        logger.warning("已拿file_title赋值")
    return item_name

@step_log("step_3_padding_item_name_to_chunks")
def step_3_padding_item_name_to_chunks(chunks, item_name):
    for chunk in chunks:
        chunk["item_name"] = item_name

@step_log("step_4_prepared_item_name_collection")
def step_4_prepared_item_name_collection():
    milvus_client = get_milvus_client()
    has_collection = milvus_client.has_collection(collection_name=milvus_config.item_name_collection)
    if has_collection:
        logger.info(f"{milvus_config.item_name_collection}存在，可以直接使用")
        return
    logger.warning(f"{milvus_config.item_name_collection}不存在，进行创建")
    schema = milvus_client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="pk",datatype=DataType.INT64,is_primary=True)
    schema.add_field(field_name="file_title",datatype=DataType.VARCHAR,max_length=512)
    schema.add_field(field_name="item_name",datatype=DataType.VARCHAR,max_length=512)
    schema.add_field(field_name="dense_vector",datatype=DataType.FLOAT_VECTOR,dim = 1024)
    schema.add_field(field_name="sparse_vector",datatype=DataType.SPARSE_FLOAT_VECTOR)
    # 3.3. Prepare index parameters
    index_params = milvus_client.prepare_index_params()

    # 3.4. Add indexes
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type = "COSINE",
        index_name="dense_vector_index", # Name of the index to create
        params={
            "M": 64, # Maximum number of neighbors each node can connect to in the graph
            "efConstruction": 100 # Number of candidate neighbors considered for connection during index construction
        }
    )

    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_vector_index",  # Name of the index to create
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
        metric_type="IP"
    )

    milvus_client.create_collection(
        collection_name=milvus_config.item_name_collection,
        schema=schema,
        index_params=index_params
    )


@step_log("step_5_insert_item_name_data")
def step_5_insert_item_name_data(item_name, file_title):
    milvus_client =  get_milvus_client()
    result = generate_embeddings([item_name])
    dense_vector = result.get("dense")[0]
    sparse_vector = result.get("sparse")[0]

    milvus_client.delete(
        collection_name=milvus_config.item_name_collection,
        filter= f"file_title == '{file_title}'"
    )
    res = milvus_client.insert(
        collection_name=milvus_config.item_name_collection,
        data = [
            {
                "item_name": item_name,
                "file_title": file_title,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
        ]
    )
    print(res)

@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    """
    add_running_task(state["task_id"], "node_item_name_recognition")
    #获取输入校验
    chunks, file_title = step_1_validate_and_get_data(state)
    # 步骤2: 使用模型提取item_name(chunks file_title) -> item_name
    item_name: str = step_2_call_llm_return_item_name(chunks, file_title)
    # 步骤3: 先修改chunks的数据 -> item_name
    step_3_padding_item_name_to_chunks(chunks, item_name)
    # for chunk in chunks:
    #     logger.success(chunk)
    # 步骤4:创建存储item_name需要的集合
    step_4_prepared_item_name_collection()
    # 步骤5: 向集合存储item_name (item_name,稠密和稀疏,file_title)
    step_5_insert_item_name_data(item_name, file_title)
    state["item_name"] = item_name
    add_done_task(state["task_id"], "node_item_name_recognition")
    return state

# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            "md_path":r"D:\dev\workspace\shopkeeper-brain-demo\output\hak180产品安全手册\hak180产品安全手册_new.md",
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "parent_title":"产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "parent_title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "parent_title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # # 4. 验证Milvus存储（可选）
        # milvus_client = get_milvus_client()
        # collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        # if milvus_client and collection_name:
        #     milvus_client.load_collection(collection_name)
        #     # 检索测试结果
        #     item_name = result_state.get('item_name')
        #     safe_name = escape_milvus_string(item_name)
        #     res = milvus_client.query(
        #         collection_name=collection_name,
        #         filter=f'item_name=="{safe_name}"',
        #         output_fields=["file_title", "item_name"]
        #     )
        #     logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()