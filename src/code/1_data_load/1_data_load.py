# 使用unstructured库来加载并且解析一个pdf文件
from unstructured.partition.auto import partition
from collections import Counter

# 设置pdf路径
pdf_path = "./data/pdf/2502.pdf"
elements = partition(filename=pdf_path, content_type="application/pdf")
# print(f"[elements]:{elements}")

types = Counter(child.category for child in elements)
print(dict(types), type(elements))
print("-" * 10)
length = len(elements)
headers_5 = [elements[i] for i in range(0, 3)]
for child in headers_5:
    print(child.category)
