def Fixed_block_size():
    """
    固定分块策略，分两步，假设每个块的最大size为max_size：
    第一步：将所有文本内容按照段落进行划分，假设划分为A、B、C、D、E和F。
    第二步：按照原有的顺序进行累加，假设A+B<max_size，A+B+C>max_size，
        那么就将B作为第一个分块，将C放置到下一个分块。
        假设E>max_size，那么E就属于超长块的警告。
    """
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.document_loaders import TextLoader

    # 创建文件读取器
    loader = TextLoader("../../data/txt/load_test.txt", encoding="utf-8")
    docs = loader.load()
    print(docs)
    print(f"[type]:{type(docs)},[len]:{len(docs)}")
    # 创建文件切分器
    text_splitter = CharacterTextSplitter(
        # 每个块的目标大小为100个字符
        chunk_size=100,
        # 每个块之间重叠10个字符，以缓解语义切割
        chunk_overlap=20,
        separator="\n",
    )
    chunks = text_splitter.split_documents(docs)
    # 输出切分之后的结果
    print(f"文本被切分为:{len(chunks)}个块")
    for i, chunk in enumerate(chunks[:3], start=1):
        print("=" * 10)
        print(
            f"[第{i}个块]-长度:{len(chunk.page_content)}\n[第{i}个块]-内容:{chunk.page_content}"
        )


def Recursive_character_chunks():
    """
    和固定分块的主流过程其实是差不多的，但关键问题就在“怎么分的问题”？
    固定分块（只能指定一个分块字符）针对大于max_size的块只能采取“保留”措施，没法进行其余的操作。
    但递归分块策略（可以指定多个分块字符），针对大于max_size的块可以递归分块字符，然后重复切割操作。
    具体流程：
    （1）寻找有效分隔符: 从分隔符列表中从前到后遍历，
    找到第一个在当前文本中存在的分隔符。如果都不存在，使用最后一个分隔符
    （通常是空字符串 ""）。
    （2）切分与分类处理: 使用选定的分隔符切分文本，然后遍历所有片段：
    如果片段不超过块大小: 暂存到 _good_splits 中，准备合并
    如果片段超过块大小:
    首先，将暂存的合格片段通过 _merge_splits 合并成块
    然后，检查是否还有剩余分隔符：
    有剩余分隔符: 递归调用 _split_text 继续分割
    无剩余分隔符: 直接保留为超长块
    （3）最终处理: 将剩余的暂存片段合并成最后的块
    实现细节：
    批处理机制: 先收集所有合格片段（_good_splits），遇到超长片段时才触发合并操作。
    递归终止条件: 关键在于 if not new_separators 判断。当分隔符用尽时（new_separators 为空），停止递归，直接保留超长片段。确保算法不会无限递归。
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader("../../data/txt/load_test.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ",", ""],
        chunk_size=200,
        chunk_overlap=20,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[chunks]:{chunks}")


def Semantic_segmentation():
    """
    语义分块：
    步骤：
    1-按照之前的规则对文本进行分块处理；
    2-上下文感知（假设存在每次a,b,c,d,e,f六个块，那么当pointer=c时，会将b-c-d串联在一起传递给嵌入模型）；
    3-计算语义距离（计算每对相邻句子的嵌入向量之间的余弦距离）；
    4-识别断点（分析所有的距离值，将大于与之的距离识别为语义上的断点）；
    5-合并成块（识别出所有的断点，然后将原始的句子序列按照断点进行切分，将切分之后的每一个部分之内的所有句子结合起来，形成一个最终语意连贯的文本块）；
    """
    from langchain_community.document_loaders import TextLoader
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # 加载文本
    loader = TextLoader("../../data/txt/load_test.txt", encoding="utf-8")
    document_obj = loader.load()
    # print(document_obj)
    # 创建语义分块器
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    text_splitter = SemanticChunker(
        # 分析器对象
        embeddings,
        # 断点识别方法
        breakpoint_threshold_type="percentile",
    )
    docs = text_splitter.split_documents(document_obj)
    print(f"[type]:{type(docs)},[len]:{len(docs)}")


def Block_segmentation_based_on_document_structure():
    """
    基于文档结构的分块
    """
    # from langchain.text_splitter import MarkdownHeaderTextSplitter
    # from langchain_community.document_loaders import ToMarkdownLoader
    pass


if __name__ == "__main__":
    # Fixed_block_size()
    # Recursive_character_chunks()
    Semantic_segmentation()
