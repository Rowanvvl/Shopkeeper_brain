# Shopkeeper_brain

把商品 / 设备的说明书文档（PDF、Markdown）自动加工成可检索的知识库：解析文档、处理图片、切片、识别主体、生成混合向量，最终写入 Milvus。整个流程用 LangGraph 编排，各节点职责单一、可单独调试和替换。

## 当前状态

项目处于**骨架已成型、节点待填充**的阶段，clone 后请先了解这一点：

| 层次 | 状态 | 说明 |
|---|---|---|
| 图编排（`processor/import_processor/main_graph.py`） | 已完成 | 7 个节点的 DAG 已连通，可运行 |
| 节点业务逻辑（`nodes/`） | **仅 `node_entry` 有实现** | 其余 6 个节点是带完整 docstring 的空函数，等着往里填具体逻辑 |
| 基础设施（`utils/`、`common/`） | 已完成 | LLM / Embedding / Milvus / MinIO / Mongo / SSE / 日志 等全部可用 |
| API 层（`api/`）、前端（`page/`） | 空目录 | 预留，尚未开始 |

节点的 docstring 里已经写清了每个节点"要做什么、输入什么、输出什么"，按注释实现即可。

## 技术栈

- **编排**：LangGraph 1.x（StateGraph + 条件边）
- **LLM**：LangChain + OpenAI 兼容接口（阿里云百炼 / 千问等）
- **向量**：BGE-M3，同时产出 Dense + Sparse 混合向量
- **存储**：Milvus（向量，支持混合检索）、MinIO（图片等对象）、MongoDB（会话历史）
- **文档解析**：MinerU / magic-pdf
- **服务**：FastAPI + Uvicorn、SSE 实时进度推送
- **环境管理**：uv（Python >= 3.12）

## 目录结构

```
Shopkeeper_brain/
├── api/                        # HTTP 接口层（预留）
├── page/                       # 前端页面（预留）
├── doc/                        # 示例文档（产品说明书 PDF）
├── test/                       # 测试（预留）
├── processor/
│   ├── import_processor/       # 知识入库管道
│   │   ├── main_graph.py       # 图装配：节点注册 + 边连接
│   │   ├── state.py            # ImportGraphState 状态定义
│   │   └── nodes/              # 7 个流程节点
│   └── query_processor/        # 检索问答管道（预留）
├── common/
│   ├── config/                 # 配置类：LLM / Embedding / Reranker / Milvus / MinIO / MinerU / MCP
│   └── logging/logger.py       # loguru 日志，含 node_log / step_log 装饰器
└── utils/
    ├── lm/                     # LLM 客户端、BGE-M3 向量化、Reranker
    ├── clients/                # Milvus / MinIO / Mongo 客户端
    ├── download_model/         # 模型下载脚本
    ├── task_utils.py           # 任务状态机（节点完成进度）
    ├── sse_utils.py            # SSE 推送（向前端推送实时进度）
    ├── rate_limit_utils.py     # API 限流
    ├── load_prompt.py          # 提示词模板加载
    └── path_util.py            # 基于 .env 定位项目根目录
```

## 快速开始

### 1. 环境要求

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)（本项目用 uv 管理依赖与虚拟环境）
- 可访问的 Milvus、MinIO、MongoDB 服务（按需，跑单个节点时不必全部就绪）

### 2. 安装依赖

```powershell
cd <项目根目录>
uv sync
```

依赖已锁定 Windows x86_64 平台的解析结果（`pyproject.toml` 中的 `required-environments`），默认走清华 PyPI 镜像。

### 3. 配置环境变量

在项目根目录创建 `.env`（该文件已被 git 忽略），参考下表填写：

```ini
# ---- 大模型（OpenAI 兼容接口）----
OPENAI_API_KEY=sk-xxxxxxxx
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_DEFAULT_MODEL=qwen3-32b
LLM_DEFAULT_TEMPERATURE=0.1
VL_MODEL=qwen3-vl-plus          # 多模态模型，用于图片理解

# ---- BGE-M3 向量模型 ----
BGE_M3=BAAI/bge-m3                      # 未指定本地路径时从此标识拉取
BGE_M3_PATH=                            # 本地模型路径，留空则自动下载
BGE_DEVICE=cpu                          # CPU 跑设 cpu；有独显设 cuda:0
BGE_FP16=0                              # CPU 下必须为 0（FP16 在 CPU 上不稳定）

# ---- Reranker（可选）----
BGE_RERANKER_LARGE=BAAI/bge-reranker-v2-m3
BGE_RERANKER_DEVICE=cpu
BGE_RERANKER_FP16=0

# ---- Milvus ----
MILVUS_URL=http://localhost:19530
CHUNKS_COLLECTION=knowledge_chunks      # 文档切片向量
ENTITY_NAME_COLLECTION=entity_names     # 预留：实体名称
ITEM_NAME_COLLECTION=item_names         # 文档对应的商品主体名

# ---- MinIO（图片等文件存储）----
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=shopkeeper
MINIO_IMG_DIR=images
MINIO_SECURE=False                      # True 表示启用 HTTPS

# ---- MongoDB（会话历史）----
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=shopkeeper_brain

# ---- MinerU（PDF 解析）----
MINERU_BASE_URL=
MINERU_API_TOKEN=

# ---- MCP（可选）----
MCP_DASHSCOPE_BASE_URL=

# ---- 日志 ----
LOG_CONSOLE_ENABLE=True
LOG_CONSOLE_LEVEL=INFO
LOG_FILE_ENABLE=True
LOG_FILE_LEVEL=DEBUG
LOG_FILE_RETENTION=7                    # 日志保留天数
```

所有配置项在各 dataclass 中有默认值或明确校验，缺失时会在调用处抛出带提示信息的异常。

### 4. 运行知识入库管道

在项目根目录下执行：

```powershell
uv run python -m processor.import_processor.main_graph
```

注意必须用 `-m` 模块方式启动。项目代码使用从根包开始的绝对导入（如 `from common.logging.logger import logger`），直接 `python xxx.py` 会因 `sys.path[0]` 变成脚本所在目录而报 `ModuleNotFoundError`。

`main_graph.py` 的 `__main__` 段会执行一次完整图调用，并把 ASCII 格式的图结构打印出来，方便确认流程是否符合预期。

## 数据流程

```
                    ┌──────────────┐
                    │     START    │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  node_entry  │  按扩展名判定 PDF / MD，设置路由标记
                    └──────┬───────┘
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌──────────────────┐      ┌────────────────┐
     │ node_pdf_to_md   │      │  node_md_img   │  PDF 解析后也汇入此节点
     │ PDF → Markdown   │─────▶│  图片 → MinIO  │
     └──────────────────┘      └───────┬────────┘
                                       ▼
                            ┌─────────────────────┐
                            │ node_document_split │  按标题层级递归切片
                            └─────────┬───────────┘
                                      ▼
                        ┌────────────────────────────┐
                        │ node_item_name_recognition │  LLM 识别文档主体商品名
                        └─────────────┬──────────────┘
                                      ▼
                            ┌─────────────────────┐
                            │ node_bge_embedding  │  Dense + Sparse 混合向量
                            └─────────┬───────────┘
                                      ▼
                            ┌─────────────────────┐
                            │ node_import_milvus  │  按商品名幂等删旧写新
                            └─────────┬───────────┘
                                      ▼
                                     END
```

### 节点与状态字段

| 节点 | 职责 | 写入的状态字段 |
|---|---|---|
| `node_entry` | 入口，按文件后缀分流 | `is_pdf_read_enabled` / `is_md_read_enabled` |
| `node_pdf_to_md` | 调用 MinerU 把 PDF 转为 Markdown | `md_content` |
| `node_md_img` | 提取文档中的图片，上传 MinIO 并回写 URL | `md_content` |
| `node_document_split` | 按标题层级切分为 Chunk，附带标题路径元数据 | `chunks` |
| `node_item_name_recognition` | 用 LLM 识别文档描述的商品主体 | `item_name` |
| `node_bge_embedding` | 为每个 Chunk 生成稠密 + 稀疏向量 | `embeddings_content` |
| `node_import_milvus` | 幂等写入 Milvus（按 `item_name` 先删后插） | — |

状态定义在 `processor/import_processor/state.py` 的 `ImportGraphState` 中，用 `create_default_state(**overrides)` 构造初始状态即可发起一次调用。

## 可复用的基础设施

这些模块已实现可直接 import 使用：

| 模块 | 关键函数 | 用途 |
|---|---|---|
| `utils/lm/lm_utils.py` | `get_llm_client(model, json_mode)` | 带缓存的 LLM 客户端；`json_mode=True` 强制 JSON 输出；适配千问等国产模型的私有参数 |
| `utils/lm/embedding_utils.py` | `generate_embeddings(texts)` | BGE-M3 混合向量，返回 `{"dense": [...], "sparse": [...]}`；NumPy 类型已转成原生 Python 类型，可直接序列化 |
| `utils/lm/reranker_utils.py` | `get_reranker_model()`、`compute_token_number()` | 重排序模型与 token 计数 |
| `utils/clients/milvus_utils.py` | `hybrid_search()`、`fetch_chunks_by_chunk_ids()` | 混合检索（稠密 + 稀疏加权）与按 ID 批量取回原文 |
| `utils/clients/minio_utils.py` | `get_minio_client()` | 获取客户端并确保桶存在 |
| `utils/clients/mongo_history_utils.py` | `save_chat_message()`、`get_recent_messages()` 等 | 多轮会话历史存取 |
| `utils/task_utils.py` | `update_task_status()`、`get_done_task_list()` 等 | 长任务进度状态机 |
| `utils/sse_utils.py` | `push_to_session()` | 按会话推送实时事件 |
| `utils/load_prompt.py` | `load_prompt(name, **kwargs)` | 加载并渲染提示词模板 |
| `utils/rate_limit_utils.py` | `apply_api_rate_limit()` | API 调用限流 |
| `common/logging/logger.py` | `logger`、`@node_log`、`@step_log` | 结构化日志，自动记录节点耗时与 trace_id |

### 模型下载

不想运行时自动拉取模型，可以预先下载到本地，再把 `BGE_M3_PATH` 指向对应目录：

```powershell
uv run python -m utils.download_model.download_bgem3
uv run python -m utils.download_model.download_reranker
```

## 环境变量速查

| 分类 | 变量 |
|---|---|
| 大模型 | `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`LLM_DEFAULT_MODEL`、`LLM_DEFAULT_TEMPERATURE`、`VL_MODEL` |
| 向量模型 | `BGE_M3`、`BGE_M3_PATH`、`BGE_DEVICE`、`BGE_FP16` |
| 重排序 | `BGE_RERANKER_LARGE`、`BGE_RERANKER_DEVICE`、`BGE_RERANKER_FP16` |
| Milvus | `MILVUS_URL`、`CHUNKS_COLLECTION`、`ENTITY_NAME_COLLECTION`、`ITEM_NAME_COLLECTION` |
| MinIO | `MINIO_ENDPOINT`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY`、`MINIO_BUCKET_NAME`、`MINIO_IMG_DIR`、`MINIO_SECURE` |
| MongoDB | `MONGO_URL`、`MONGO_DB_NAME` |
| MinerU | `MINERU_BASE_URL`、`MINERU_API_TOKEN` |
| MCP | `MCP_DASHSCOPE_BASE_URL` |
| 日志 | `LOG_CONSOLE_ENABLE`、`LOG_CONSOLE_LEVEL`、`LOG_FILE_ENABLE`、`LOG_FILE_LEVEL`、`LOG_FILE_RETENTION` |

## 开发注意事项

- **导入方式**：统一使用从项目根包出发的绝对导入，运行脚本一律 `-m` 方式。
- **CPU / GPU**：CPU 环境下 `BGE_FP16` 必须设为 `0`，否则半精度推理会出现异常结果或报错。
- **日志**：默认写入 `logs/` 目录，按天切分并按 `LOG_FILE_RETENTION` 自动清理。
- **幂等性**：入库设计上按 `item_name` 先删除旧数据再写入，避免重复导入造成向量堆积。

## 路线图

- 实现 `node_pdf_to_md`：接入 MinerU 完成 PDF → Markdown
- 实现 `node_md_img`：图片上传 MinIO + 多模态描述
- 实现 `node_document_split`：基于标题层级的递归切片
- 实现 `node_item_name_recognition`：LLM 主体识别
- 实现 `node_bge_embedding` 与 `node_import_milvus`：完成入库闭环
- `query_processor`：检索问答管道
- `api/`：对外 HTTP 接口；`page/`：交互页面
