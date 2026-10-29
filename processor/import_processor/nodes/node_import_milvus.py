import sys

from pymilvus import MilvusClient, has_collection, DataType

from common.config.milvus_config import milvus_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.clients.milvus_utils import get_milvus_client
from utils.task_utils import add_running_task, add_done_task
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    embeddings_content = state.get("embeddings_content")
    if not embeddings_content:
        logger.error("embedding没有值，业务提前终止")
        raise ValueError("embedding没有值，业务提前终止")

    return embeddings_content

@step_log("step_2_prepared_item_name_collection")
def step_2_prepared_item_name_collection():
    milvus_client = get_milvus_client()

    has_collection = milvus_client.has_collection(collection_name=CHUNKS_COLLECTION_NAME)
    if has_collection:
        logger.success("集合存在，直接使用")
        return
    logger.warning("集合不存在需要创建")

    schema = milvus_client.create_schema(
        auto_id = True,
        enable_dynamic_field = True
    )

    schema.add_field(field_name = "chunk_id", datatype = DataType.INT64,is_primary = True)
    schema.add_field(field_name = "content", datatype = DataType.VARCHAR,max_length = 65535)
    schema.add_field(field_name = "file_title", datatype = DataType.VARCHAR,max_length = 512)
    schema.add_field(field_name = "item_name", datatype = DataType.VARCHAR,max_length = 512)
    schema.add_field(field_name = "title", datatype = DataType.VARCHAR,max_length = 512)
    schema.add_field(field_name = "parent_title", datatype = DataType.VARCHAR,max_length = 512)
    schema.add_field(field_name = "part", datatype = DataType.INT8)
    schema.add_field(field_name = "dense_vector", datatype = DataType.FLOAT_VECTOR,dim = 1024)
    schema.add_field(field_name = "sparse_vector", datatype = DataType.SPARSE_FLOAT_VECTOR)

    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        index_name="dense_vector_index",
        params={
            "M": 64,  # 相邻最大的节点数量
            "efConstruction": 100  # 候选的节点数量
        },
        metric_type = "COSINE"
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_vector_index",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
        metric_type = "IP"
    )
    milvus_client.create_collection(collection_name=CHUNKS_COLLECTION_NAME,schema=schema,index_params=index_params)

    milvus_client.load_collection(collection_name=CHUNKS_COLLECTION_NAME)

@step_log("step_3_insert_item_name_data")
def step_3_insert_item_name_data(embeddings_content):
    file_title = embeddings_content[0].get("file_title")
    milvus_client=get_milvus_client()
    milvus_client.delete(
        collection_name=milvus_config.chunks_collection,
        filter= f"file_title == '{file_title}'"
    )
    milvus_client.insert(
        collection_name=milvus_config.chunks_collection,
        data=embeddings_content
    )

@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    """
    add_running_task(state["task_id"], "node_import_milvus")
    # 1.获取并校验参数
    embeddings_content = step_1_validate_and_get_data(state)
    # 2. 准备存储切片集合
    step_2_prepared_item_name_collection()
    # 3. 插入数据的函数
    step_3_insert_item_name_data(embeddings_content)
    logger.info(
        f"{embeddings_content[0].get("file_title")}文档对应的数据已经完成向量数据库的导入,导入的数量:{len(embeddings_content)}!")
    add_done_task(state["task_id"], "node_import_milvus")
    return state

if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "item_name":"测试项目_Milvus",
        "file_title": "test.pdf",
        "embeddings_content": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
,
            {
                "content": "Milvus 测试文本 2",
                "title": "测试标题2",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title": "test.pdf2",
                "part": 1,
                "file_title": "test.pdf",
                "dense_vector": [0.2] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 执行节点函数
        result_state = node_import_milvus(test_state)
    except Exception as e:
        print(f"❌ 测试失败: {e}")