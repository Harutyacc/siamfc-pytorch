import torch
from torchviz import make_dot
from siamfc.backbones import AlexNetV1
from siamfc.heads import SiamFC

# 定义示例输入数据
z = torch.randn(1, 3, 127, 127)  # 模板图像
x = torch.randn(1, 3, 255, 255)  # 搜索图像

# 初始化 AlexNetV1 网络
alexnet_v1 = AlexNetV1()
# 获取 AlexNetV1 网络的输出
z_feat = alexnet_v1(z)
x_feat = alexnet_v1(x)

# 初始化 SiamFC 网络
siamfc = SiamFC(out_scale=0.001)
# 获取 SiamFC 网络的输出
output = siamfc(z_feat, x_feat)

# # 生成 AlexNetV1 网络结构图
# alexnet_v1_graph = make_dot(z_feat, params=dict(list(alexnet_v1.named_parameters())))
# alexnet_v1_graph.render("alexnet_v1_structure", format="png")

# # 生成 SiamFC 网络结构图
# siamfc_graph = make_dot(output, params=dict(list(siamfc.named_parameters())))
# siamfc_graph.render("siamfc_structure", format="png")

# 生成 AlexNetV1 网络结构图并保存为 dot 文件
alexnet_v1_graph = make_dot(z_feat, params=dict(list(alexnet_v1.named_parameters())))
alexnet_v1_graph.save("fig/alexnet_v1_structure.dot")

# 生成 SiamFC 网络结构图并保存为 dot 文件
siamfc_graph = make_dot(output, params=dict(list(siamfc.named_parameters())))
siamfc_graph.save("fig/siamfc_structure.dot")

print("网络结构图已生成并保存为 alexnet_v1_structure.dot 和 siamfc_structure.dot")

# 生成高分辨率 PNG 图像
import os
os.system('dot -Tpng -Gdpi=300 fig/alexnet_v1_structure.dot -o fig/alexnet_v1_structure.png')
os.system('dot -Tpng -Gdpi=300 fig/siamfc_structure.dot -o fig/siamfc_structure.png')

print("高分辨率网络结构图已生成并保存为 alexnet_v1_structure.png 和 siamfc_structure.png")
