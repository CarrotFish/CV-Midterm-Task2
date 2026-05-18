import matplotlib.pyplot as plt

def parse_yolo_csv(file_path):
    """
    手动解析 YOLOv8 的 results.csv 文件
    不使用 pandas，纯 Python 字符串处理
    """
    data = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None

    if not lines:
        print("错误：CSV 文件为空")
        return None

    # 1. 解析表头 (去除 YOLOv8 CSV 表头中常见的首尾空格)
    headers = [header.strip() for header in lines[0].split(',')]
    
    # 初始化数据字典，每个表头对应一个空列表
    for header in headers:
        data[header] = []

    # 2. 逐行解析数据
    for line in lines[1:]:
        line = line.strip()
        if not line:  # 跳过空行
            continue
            
        values = line.split(',')
        
        for i, val in enumerate(values):
            header = headers[i]
            # 去除数值前后的空格，并转换为浮点数
            clean_val = val.strip()
            try:
                # 处理可能为空或非数字的单元格
                data[header].append(float(clean_val))
            except ValueError:
                # 如果转换失败（比如有异常字符串），填入 None
                data[header].append(None)
                
    return data

def plot_training_curves(data):
    """
    可视化训练曲线，分为损失、指标和学习率三个核心部分
    """
    if not data or 'epoch' not in data:
        print("无效的数据，无法绘图。")
        return

    epochs = data['epoch']

    # 创建一个 2 行 3 列的画布，尺寸为 15x10
    plt.figure(figsize=(16, 10))
    plt.suptitle("YOLOv8 Training Curves", fontsize=16, fontweight='bold')

    # ================= 1. Box Loss (训练集 vs 验证集) =================
    plt.subplot(2, 3, 1)
    plt.plot(epochs, data['train/box_loss'], label='Train Box Loss', color='blue', linewidth=2)
    plt.plot(epochs, data['val/box_loss'], label='Val Box Loss', color='orange', linewidth=2, linestyle='--')
    plt.title('Box Loss (CIoU)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # ================= 2. Class Loss (训练集 vs 验证集) =================
    plt.subplot(2, 3, 2)
    plt.plot(epochs, data['train/cls_loss'], label='Train Cls Loss', color='green', linewidth=2)
    plt.plot(epochs, data['val/cls_loss'], label='Val Cls Loss', color='red', linewidth=2, linestyle='--')
    plt.title('Classification Loss (BCE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # ================= 3. DFL Loss (训练集 vs 验证集) =================
    plt.subplot(2, 3, 3)
    plt.plot(epochs, data['train/dfl_loss'], label='Train DFL Loss', color='purple', linewidth=2)
    plt.plot(epochs, data['val/dfl_loss'], label='Val DFL Loss', color='brown', linewidth=2, linestyle='--')
    plt.title('Distribution Focal Loss (DFL)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # ================= 4. 评估指标 (Metrics) =================
    plt.subplot(2, 3, 4)
    plt.plot(epochs, data['metrics/precision(B)'], label='Precision', color='cyan')
    plt.plot(epochs, data['metrics/recall(B)'], label='Recall', color='magenta')
    plt.plot(epochs, data['metrics/mAP50(B)'], label='mAP@0.5', color='red', linewidth=2.5)
    plt.plot(epochs, data['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='blue', linewidth=2.5)
    plt.title('Metrics (Precision, Recall, mAP)')
    plt.xlabel('Epoch')
    plt.ylabel('Value (0-1)')
    plt.ylim(0, 1.05) # 指标范围通常在 0-1 之间
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # ================= 5. 学习率 (Learning Rate) =================
    plt.subplot(2, 3, 5)
    plt.plot(epochs, data['lr/pg0'], label='lr/pg0 (Bias)', color='gray')
    plt.plot(epochs, data['lr/pg1'], label='lr/pg1 (Weight)', color='black')
    plt.plot(epochs, data['lr/pg2'], label='lr/pg2 (Weight w/ Decay)', color='olive', linestyle='-.')
    plt.title('Learning Rates')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # 调整子图之间的间距，防止重叠
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 显示图像
    plt.show()

if __name__ == '__main__':
    csv_file_path = 'results.csv' 
    
    print("正在解析 CSV 文件...")
    parsed_data = parse_yolo_csv(csv_file_path)
    
    if parsed_data:
        print("解析成功，正在绘制曲线...")
        plot_training_curves(parsed_data)