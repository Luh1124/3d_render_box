
import bpy  
import sys  

# 加载blend文件路径从命令行参数传入  
blend_path = sys.argv[-1]  

# 打开指定的.blend文件（以覆盖当前场景方式打开）  
bpy.ops.wm.open_mainfile(filepath=blend_path)  

# 遍历所有图像，打印名称和路径  
for img in bpy.data.images:  
    print(f"Image Name: {img.name}, Filepath: {img.filepath}")  