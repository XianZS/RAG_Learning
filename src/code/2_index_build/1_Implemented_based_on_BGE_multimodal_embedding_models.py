# 基于BGE的多模态嵌入模型的实现
# import os

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE

model = Visualized_BGE(
    # 指明模型的名称
    model_name_bge="BAAI/bge-base-en-v1.5",
    # 指明模型的位置
    model_weight="./models/bge/Visualized_base_en_v1.5.pth",
)
with torch.no_grad():
    """
        --- 训练阶段：
        在这个代码之中，首先创建文本向量，将文本映射到向量空间之中，得到text_emb
        然后分别创建两组训练数据，分别为(img_1,text)和(img_2,text),将img_1和img_2也映射到相同的向量空间之中
        最后分别得到两个训练张量img_emb_1和img_emb_2。
        --- 验证阶段：
        在验证阶段，我们可以导入几张照片，验证图片包括“正确验证图片集合”和“错误验证照片集合”。
    """
    # print("===" * 10, "训练阶段", "===" * 10)
    text_emb = model.encode(text="datawhale开源组织的logo")
    img_emb_1 = model.encode(
        image="./imgs/datawhale01.png", text="datawhale开源组织的logo"
    )
    img_emb_2 = model.encode(
        image="./imgs/datawhale02.png", text="datawhale开源组织的logo"
    )
    # 得到本地训练张量集合
    embs = [img_emb_1, img_emb_2]
    # print("===" * 10, "验证阶段", "===" * 10)


def ver_func(ver_obj):
    # print("-" * 60)
    # ver_obj：是你需要验证的对象
    # 计算每一张“纯图片”和“图片文本”的相似度
    # print(f"[image @ plain text]:\n{ver_obj @ text_emb.T}")  # type:ignore
    sim_1 = ver_obj @ img_emb_1.T  # type:ignore
    sim_2 = ver_obj @ img_emb_2.T  # type:ignore
    # print(f"[image @ Combining text and images 1]:\n{sim_1}")  # type:ignore
    # print(f"[image @ Combining text and images 2]:\n{sim_2}")  # type:ignore
    # print("-" * 60)
    if sim_1 > 0.70 and sim_2 > 0.70:
        # 得到两张“图片文本”和“纯文本”相似度计算的平均数值
        average = img_emb_1 @ text_emb.T + img_emb_2 @ text_emb.T  # type:ignore
        print(f"[average]:{average[0][0]}")
        if average[0][0] / 2 > 0.65:
            return True
    return False


print("=" * 30, "OUTPUT", "=" * 30)
need_vers = []
# 加载3张正确的图片
with torch.no_grad():
    img_1 = model.encode("./imgs/Correctly verify Image-1.png")
    img_2 = model.encode("./imgs/Correctly verify Image-2.png")
    img_3 = model.encode("./imgs/Correctly verify Image-3.png")
    need_vers.append(img_1)
    need_vers.append(img_2)
    need_vers.append(img_3)

# 加载3张错误的图片
with torch.no_grad():
    img_1 = model.encode("./imgs/Error verification image -1.jpg")
    img_2 = model.encode("./imgs/Error verification image -2.png")
    img_3 = model.encode("./imgs/Error verification image -3.png")
    need_vers.append(img_1)
    need_vers.append(img_2)
    need_vers.append(img_3)
print(f"[len-need_vers]:{len(need_vers)}")
print("-" * 60)
index = 1
for child in need_vers:
    print(f"[index={index}]:>>>")
    judge = ver_func(child)
    print(f"[judge]={judge}")
    index += 1
    print("-" * 30)

"""
[结论]：
    - 正确-2和正确-3这两张图片和本地训练集合之中的图片内容几乎是匹配的，相似度差不多在75%以上。
    - 但是几乎所有照片和纯图片的相似度都在40%~50%之间，也就意味着，
      如果只是单纯的将图片和纯文本映射到同一个维度空间之中，
      就几乎起不到任何的分类效果
    - 正确的做法时，将“纯图片”和“图片文本”进行相似度计算，
      然后将“纯文本”和“图片文本”进行相似度计算，
      最后根据计算结果就可以得到图片对应的文本。
    - “图片文本”就起到一个连接器的作用
============================== OUTPUT ==============================
[len-need_vers]:6
------------------------------------------------------------
[index=1]:>>>
------------------------------------------------------------
[image @ plain text]:
tensor([[0.4998]])
[image @ Combining text and images 1]:
tensor([[0.5551]])
[image @ Combining text and images 2]:
tensor([[0.5446]])
------------------------------------------------------------
[index=2]:>>>
------------------------------------------------------------
[image @ plain text]:
tensor([[0.4624]])
[image @ Combining text and images 1]:
tensor([[0.7949]])
[image @ Combining text and images 2]:
tensor([[0.7510]])
------------------------------------------------------------
[index=3]:>>>
------------------------------------------------------------
[image @ plain text]:
tensor([[0.5726]])
[image @ Combining text and images 1]:
tensor([[0.7971]])
[image @ Combining text and images 2]:
tensor([[0.7512]])
------------------------------------------------------------
[index=4]:>>>
------------------------------------------------------------
[image @ plain text]:
tensor([[0.4536]])
[image @ Combining text and images 1]:
tensor([[0.4520]])
[image @ Combining text and images 2]:
tensor([[0.4836]])
------------------------------------------------------------
[index=5]:>>>
------------------------------------------------------------
[image @ plain text]:
tensor([[0.4661]])
[image @ Combining text and images 1]:
tensor([[0.5026]])
[image @ Combining text and images 2]:
tensor([[0.4708]])
------------------------------------------------------------
[index=6]:>>>
------------------------------------------------------------
[image @ plain text]:
tensor([[0.4245]])
[image @ Combining text and images 1]:
tensor([[0.5289]])
[image @ Combining text and images 2]:
tensor([[0.4652]])
------------------------------------------------------------
"""
