# RAG_Learning
RAG_Learning

## 一、深度学习基础框架与加速
这是所有AI模型运行的底层基础设施，为嵌入生成、LLM推理提供计算支持。

| 包名 | 主要作用 |
|------|----------|
| `torch==2.6.0` | **核心深度学习框架**，提供张量计算、自动微分和GPU加速能力，是所有PyTorch生态模型的运行基础 |
| `torchvision==0.21.0` | PyTorch的计算机视觉库，处理图像数据，在多模态RAG中用于图像特征提取 |
| `torchaudio==2.6.0` | PyTorch的音频处理库，在语音RAG中用于音频转文字和特征提取 |
| `accelerate>=0.20.0` | **Hugging Face官方加速工具**，简化多GPU/TPU分布式训练和推理，自动处理设备分配和数据并行 |
| `einops==0.8.1` | 优雅的张量操作库，简化复杂的张量维度变换，在模型代码中广泛使用 |

## 二、Hugging Face生态（模型与数据核心）
RAG系统中**嵌入模型**和**开源LLM**的主要来源，提供了统一的模型加载和数据处理接口。

| 包名 | 主要作用 |
|------|----------|
| `huggingface-hub>=0.33.4` | Hugging Face Hub客户端，用于下载/上传模型、数据集和空间，管理模型版本 |
| `transformers>=4.40.0` | **核心包**，提供数千种预训练模型（BERT、GPT、LLaMA等）的统一API，支持文本生成、分类、翻译等任务 |
| `tokenizers>=0.19.0` | 高性能分词器库，由Rust实现，比纯Python分词器快数十倍，是transformers的底层依赖 |
| `sentence-transformers>=3.0.0` | **RAG核心包**，专门用于生成句子/段落级别的向量嵌入，提供多种预训练嵌入模型（如all-MiniLM-L6-v2） |
| `datasets>=2.14.0` | 数据集处理库，支持加载数百种公开数据集，也可用于加载和处理自定义RAG知识库数据 |

## 三、RAG编排框架（两大主流）
这是RAG系统的"骨架"，将数据加载、分割、嵌入、检索、生成等步骤串联成完整流程。

### 3.1 LangChain生态
| 包名 | 主要作用 |
|------|----------|
| `langchain==0.3.26` | **LangChain主包**，提供RAG、Agent、Chain等核心抽象和组件 |
| `langchain-core==0.3.71` | LangChain核心库，定义了所有基础接口（如LLM、Embeddings、VectorStore），是其他LangChain包的依赖 |
| `langchain-community==0.3.27` | 社区贡献的集成组件，包含大量第三方工具、向量数据库和模型的封装 |
| `langchain-huggingface==0.3.1` | Hugging Face生态与LangChain的集成，支持在LangChain中使用Hugging Face的LLM和嵌入模型 |
| `langchain-openai==0.3.28` | OpenAI API与LangChain的集成，支持GPT系列模型和OpenAI嵌入 |
| `langchain-text-splitters==0.3.8` | **RAG核心包**，提供多种文本分割策略（字符分割、递归字符分割、语义分割等） |
| `langchain-experimental==0.3.4` | 实验性功能集合，包含一些尚未稳定的高级RAG技术（如Self-RAG、Graph RAG） |
| `langchain-deepseek==0.1.4` | DeepSeek大模型与LangChain的集成 |

### 3.2 LlamaIndex生态
| 包名 | 主要作用 |
|------|----------|
| `llama-index==0.12.51` | **LlamaIndex主包**，专注于数据索引和检索，提供更灵活的索引结构和查询引擎 |
| `llama-index-core==0.12.52` | LlamaIndex核心库，定义了索引、节点、查询引擎等基础抽象 |
| `llama-index-embeddings-huggingface==0.5.5` | 在LlamaIndex中使用Hugging Face嵌入模型 |
| `llama-index-embeddings-openai==0.3.1` | 在LlamaIndex中使用OpenAI嵌入模型 |
| `llama-index-llms-openai==0.4.7` | 在LlamaIndex中使用OpenAI LLM |
| `llama-index-llms-deepseek==0.1.2` | 在LlamaIndex中使用DeepSeek大模型 |
| `llama-index-experimental==0.5.6` | LlamaIndex的实验性功能，包含高级RAG技术 |

## 四、向量存储与检索（RAG核心组件）
负责存储嵌入向量并提供高效的相似性检索能力，是RAG系统性能的关键。

| 包名 | 主要作用 |
|------|----------|
| `chromadb>=0.4.0` | **轻量级嵌入式向量数据库**，无需独立部署，适合快速原型开发和小型应用 |
| `faiss-cpu>=1.7.0` | Facebook开发的**高性能向量检索库**，提供多种索引算法（如Flat、IVF、HNSW），CPU版本适合本地部署 |
| `pymilvus==2.5.11` | Milvus分布式向量数据库的Python客户端，适合大规模生产环境 |
| `pymilvus.model==0.3.2` | Milvus的模型集成工具，支持直接在Milvus中使用常见的嵌入模型 |

## 五、数据加载与解析（RAG数据预处理）
负责将各种格式的非结构化数据转换为可处理的文本格式，是RAG系统的"入口"。

| 包名 | 主要作用 |
|------|----------|
| `pypdf>=4.0.0` | 纯Python PDF解析库，用于提取PDF中的文本内容 |
| `unstructured==0.18.11` | **非结构化数据处理核心包**，支持解析PDF、Word、Excel、PPT、图片、邮件等几乎所有格式的文档 |
| `unstructured-client==0.41.0` | Unstructured API客户端，用于调用云端的文档解析服务 |
| `unstructured-inference==1.0.5` | Unstructured的本地推理支持，提供OCR、表格识别等能力 |
| `unstructured.pytesseract==0.3.15` | Unstructured的OCR集成，用于提取扫描版PDF和图片中的文字 |
| `unstructured[pdf]` | Unstructured的PDF解析扩展包，包含所有PDF处理所需的依赖 |
| `Markdown==3.8.2` | Markdown文档解析库 |
| `openpyxl==3.1.5` | Excel文件解析库，支持.xlsx格式 |
| `opencv-python-headless==4.12.0.88` | 计算机视觉库，用于图像处理和OCR预处理 |
| `bilibili-api-python==17.3.0` | B站API客户端，可用于获取B站视频的字幕和元数据，构建视频RAG |

## 六、大语言模型API客户端
用于调用云端大语言模型服务，生成最终的回答。

| 包名 | 主要作用 |
|------|----------|
| `openai>=1.86.0,<2.0.0` | **OpenAI官方API客户端**，支持调用GPT-3.5、GPT-4、GPT-4o等模型 |

## 七、基础工具与通用依赖
提供数据处理、配置管理、进度显示等通用功能，是所有Python项目的常用依赖。

| 包名 | 主要作用 |
|------|----------|
| `numpy>=1.24.0` | 数值计算基础库，提供数组操作和数学函数 |
| `pandas>=2.0.0` | 数据处理和分析库，用于处理表格数据 |
| `scikit-learn>=1.3.0` | 机器学习库，提供数据预处理、聚类、分类等算法，在RAG中可用于数据清洗和评估 |
| `scipy>=1.10.0` | 科学计算库，提供高级数学函数和统计工具 |
| `python-dotenv>=1.0.0` | 从.env文件加载环境变量，用于管理API密钥等敏感信息 |
| `requests>=2.28.0` | HTTP请求库，用于调用各种Web API |
| `tqdm>=4.64.0` | 进度条库，在数据处理和模型推理时显示进度 |
| `pydantic>=2.0.0` | 数据验证和设置管理库，用于定义数据模型和验证输入输出 |
| `ftfy==6.3.1` | 文本修复库，自动修复乱码、错误编码等文本问题 |
| `lark==1.2.2` | 解析器生成器，用于解析自定义语法 |
| `lazy_loader==0.4` | 延迟加载工具，优化模块导入速度 |
| `pyarrow==20.0.0` | 列式数据格式库，用于高效的数据存储和传输，是datasets和pandas的依赖 |

