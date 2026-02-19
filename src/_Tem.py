# app.py
import gradio as gr
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
from PIL import Image

# 模型定义（同之前）
class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super().__init__()
        layers = []
        prev_size = input_dim
        
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(torch.nn.Linear(prev_size, hidden_size))
            layers.append(torch.nn.BatchNorm1d(hidden_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(torch.nn.Linear(prev_size, 1))
        self.network = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_model(
    learning_rate,
    batch_size,
    epochs,
    dropout_rate,
    hidden_layers_str,
    loss_function,
    optimizer_type
):
    """训练模型并返回结果"""
    try:
        # 解析隐藏层
        hidden_layers = [int(x.strip()) for x in hidden_layers_str.split(',') if x.strip()]
        
        # 加载数据
        file_path = './datasets/housingPrice/data.xls'
        df = pd.read_csv(file_path)
        df = df.drop(columns=['date', 'street', 'city', 'statezip', 'country'])
        X = df.drop(columns=['price']).values
        y = df['price'].values.reshape(-1, 1)
        
        # 数据预处理
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = scaler_x.fit_transform(X_train)
        X_val_scaled = scaler_x.transform(X_val)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        
        # 创建模型
        model = Net(X_train.shape[1], hidden_layers, dropout_rate)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # 定义损失函数
        if loss_function == "MSE":
            criterion = torch.nn.MSELoss()
        elif loss_function == "Huber":
            criterion = torch.nn.HuberLoss(delta=1.0)
        else:  # MAE
            criterion = torch.nn.L1Loss()
        
        # 定义优化器
        if optimizer_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        else:  # SGD
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        
        # 训练循环
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0
            for i in range(0, len(X_train_scaled), batch_size):
                batch_x = torch.FloatTensor(X_train_scaled[i:i+batch_size]).to(device)
                batch_y = torch.FloatTensor(y_train_scaled[i:i+batch_size]).to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / (len(X_train_scaled) // batch_size + 1)
            train_losses.append(avg_train_loss)
            
            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i in range(0, len(X_val_scaled), batch_size):
                    batch_x = torch.FloatTensor(X_val_scaled[i:i+batch_size]).to(device)
                    batch_y = torch.FloatTensor(y_val_scaled[i:i+batch_size]).to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / (len(X_val_scaled) // batch_size + 1)
            val_losses.append(avg_val_loss)
        
        # 绘制损失曲线
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label='Train Loss', linewidth=2)
        ax.plot(val_losses, label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        
        # 保存图像
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        # 返回最终结果和图像
        result_text = f"""
        ✅ 训练完成！
        
        📊 最终结果：
        - 最终训练损失: {train_losses[-1]:.4f}
        - 最终验证损失: {val_losses[-1]:.4f}
        - 最佳验证损失: {min(val_losses):.4f} (epoch {val_losses.index(min(val_losses)) + 1})
        
        ⚙️ 使用参数：
        - 学习率: {learning_rate}
        - 批次大小: {batch_size}
        - 训练轮数: {epochs}
        - Dropout率: {dropout_rate}
        - 隐藏层: {hidden_layers}
        - 损失函数: {loss_function}
        - 优化器: {optimizer_type}
        """
        
        return result_text, Image.open(buf)
        
    except Exception as e:
        return f"❌ 训练出错: {str(e)}", None

# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft(), title="房价预测模型调试器") as demo:
    gr.Markdown("# 🏠 房价预测模型调试器")
    gr.Markdown("调整以下参数来训练你的神经网络模型")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 参数输入
            learning_rate = gr.Slider(
                minimum=0.00001, maximum=0.01, value=0.001, step=0.0001,
                label="学习率 (Learning Rate)", info="建议范围: 0.0001 - 0.001"
            )
            
            batch_size = gr.Slider(
                minimum=16, maximum=256, value=64, step=16,
                label="批次大小 (Batch Size)", info="16, 32, 64, 128, 256"
            )
            
            epochs = gr.Slider(
                minimum=10, maximum=200, value=50, step=10,
                label="训练轮数 (Epochs)", info="训练迭代次数"
            )
            
            dropout_rate = gr.Slider(
                minimum=0.0, maximum=0.5, value=0.1, step=0.05,
                label="Dropout率", info="防止过拟合，0-0.5"
            )
            
            hidden_layers = gr.Textbox(
                label="隐藏层结构",
                value="128, 64, 32",
                info="用逗号分隔，例如: 128, 64, 32"
            )
            
            loss_function = gr.Dropdown(
                choices=["MSE", "Huber", "MAE"], 
                value="Huber",
                label="损失函数"
            )
            
            optimizer = gr.Dropdown(
                choices=["Adam", "AdamW", "SGD"], 
                value="AdamW",
                label="优化器"
            )
            
            train_btn = gr.Button("🚀 开始训练", variant="primary", size="lg")
            
        with gr.Column(scale=2):
            # 输出区域
            output_text = gr.Textbox(
                label="训练结果",
                lines=15,
                interactive=False
            )
            
            output_image = gr.Image(
                label="训练曲线",
                type="pil"
            )
    
    # 示例参数按钮
    with gr.Row():
        gr.Examples(
            examples=[
                [0.001, 64, 50, 0.1, "128, 64, 32", "Huber", "AdamW"],
                [0.0005, 32, 100, 0.2, "256, 128, 64, 32", "MSE", "Adam"],
                [0.01, 128, 30, 0.0, "64, 32", "MAE", "SGD"],
            ],
            inputs=[learning_rate, batch_size, epochs, dropout_rate, hidden_layers, loss_function, optimizer],
            label="示例参数配置"
        )
    
    # 绑定事件
    train_btn.click(
        fn=train_model,
        inputs=[learning_rate, batch_size, epochs, dropout_rate, hidden_layers, loss_function, optimizer],
        outputs=[output_text, output_image]
    )
    
    # 添加说明
    gr.Markdown("""
    ### 📝 使用说明
    1. 调整左侧参数或使用示例配置
    2. 点击"开始训练"按钮
    3. 查看右侧的训练结果和损失曲线
    
    ### 💡 参数建议
    - **学习率**: 小数值更稳定，大数值收敛快但可能震荡
    - **批次大小**: 太小训练慢，太大可能内存不足
    - **Dropout**: 0.1-0.3通常效果不错
    - **隐藏层**: 从简单开始，逐步增加复杂度
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)