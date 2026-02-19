# streamlit_app.py
import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 页面配置
st.set_page_config(
    page_title="房价预测模型调试器",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题
st.title("🏠 房价预测模型调试器")
st.markdown("实时调整参数并训练神经网络模型")

# 侧边栏 - 参数设置
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # 模型参数
    st.subheader("模型结构")
    hidden_layers = st.text_input(
        "隐藏层结构（用逗号分隔）",
        value="128,64,32",
        help="例如：128,64,32 或 256,128,64,32"
    )
    
    dropout_rate = st.slider(
        "Dropout率", 
        min_value=0.0, 
        max_value=0.5, 
        value=0.1, 
        step=0.05
    )
    
    # 训练参数
    st.subheader("训练参数")
    learning_rate = st.number_input(
        "学习率", 
        min_value=1e-5, 
        max_value=0.1, 
        value=0.001, 
        format="%.5f"
    )
    
    batch_size = st.select_slider(
        "批次大小",
        options=[16, 32, 64, 128, 256],
        value=64
    )
    
    epochs = st.slider("训练轮数", 10, 200, 50)
    
    # 其他参数
    st.subheader("其他设置")
    loss_function = st.selectbox(
        "损失函数",
        ["MSE", "Huber", "MAE"],
        index=1
    )
    
    optimizer_type = st.selectbox(
        "优化器",
        ["Adam", "AdamW", "SGD"],
        index=1
    )
    
    # 示例配置
    st.subheader("示例配置")
    if st.button("快速配置 A（标准）"):
        st.session_state.lr = 0.001
        st.session_state.batch = 64
        st.session_state.epochs = 50
        st.session_state.dropout = 0.1
        st.session_state.layers = "128,64,32"
        
    if st.button("快速配置 B（深度网络）"):
        st.session_state.lr = 0.0005
        st.session_state.batch = 32
        st.session_state.epochs = 100
        st.session_state.dropout = 0.2
        st.session_state.layers = "256,128,64,32"
    
    # 开始训练按钮
    train_button = st.button("🚀 开始训练", type="primary", use_container_width=True)

# 主区域
col1, col2 = st.columns([2, 1])

with col1:
    # 训练结果区域
    if train_button:
        with st.spinner("正在训练模型，请稍候..."):
            try:
                # 解析参数
                hidden_layers_list = [int(x.strip()) for x in hidden_layers.split(',') if x.strip()]
                
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
                class Net(torch.nn.Module):
                    def __init__(self, input_dim, hidden_layers, dropout_rate):
                        super().__init__()
                        layers = []
                        prev_size = input_dim
                        
                        for hidden_size in hidden_layers:
                            layers.append(torch.nn.Linear(prev_size, hidden_size))
                            layers.append(torch.nn.BatchNorm1d(hidden_size))
                            layers.append(torch.nn.ReLU())
                            layers.append(torch.nn.Dropout(dropout_rate))
                            prev_size = hidden_size
                        
                        layers.append(torch.nn.Linear(prev_size, 1))
                        self.network = torch.nn.Sequential(*layers)
                    
                    def forward(self, x):
                        return self.network(x)
                
                model = Net(X_train.shape[1], hidden_layers_list, dropout_rate)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                
                # 训练过程
                if loss_function == "MSE":
                    criterion = torch.nn.MSELoss()
                elif loss_function == "Huber":
                    criterion = torch.nn.HuberLoss(delta=1.0)
                else:
                    criterion = torch.nn.L1Loss()
                
                if optimizer_type == "Adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                elif optimizer_type == "AdamW":
                    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
                
                # 训练循环
                train_losses = []
                val_losses = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for epoch in range(epochs):
                    # 更新进度
                    progress_bar.progress((epoch + 1) / epochs)
                    status_text.text(f"训练中... Epoch {epoch + 1}/{epochs}")
                    
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
                
                # 显示结果
                status_text.text("✅ 训练完成！")
                
                # 创建图表
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # 损失曲线
                ax1.plot(train_losses, label='训练损失', linewidth=2)
                ax1.plot(val_losses, label='验证损失', linewidth=2)
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('训练过程')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 最终损失对比
                x_pos = [0, 1]
                losses = [train_losses[-1], val_losses[-1]]
                bars = ax2.bar(x_pos, losses, color=['#1f77b4', '#ff7f0e'])
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(['训练损失', '验证损失'])
                ax2.set_ylabel('Loss')
                ax2.set_title('最终损失对比')
                
                # 在柱状图上显示数值
                for bar, loss in zip(bars, losses):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                            f'{loss:.4f}', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 显示训练摘要
                with col2:
                    st.success("训练完成！")
                    
                    st.metric("最终训练损失", f"{train_losses[-1]:.4f}")
                    st.metric("最终验证损失", f"{val_losses[-1]:.4f}")
                    
                    best_epoch = val_losses.index(min(val_losses)) + 1
                    st.metric("最佳验证损失", f"{min(val_losses):.4f}", f"Epoch {best_epoch}")
                    
                    # 显示使用的参数
                    with st.expander("📋 查看使用的参数"):
                        st.write(f"**学习率**: {learning_rate}")
                        st.write(f"**批次大小**: {batch_size}")
                        st.write(f"**训练轮数**: {epochs}")
                        st.write(f"**Dropout率**: {dropout_rate}")
                        st.write(f"**隐藏层**: {hidden_layers_list}")
                        st.write(f"**损失函数**: {loss_function}")
                        st.write(f"**优化器**: {optimizer_type}")
                        
            except Exception as e:
                st.error(f"训练出错: {str(e)}")
    else:
        # 初始状态
        st.info("👈 在左侧设置参数，然后点击'开始训练'按钮")
        
        # 显示示例图表
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.linspace(0, 50, 100)
        ax.plot(x, 10 * np.exp(-x/10) + np.random.normal(0, 0.2, 100), label='训练损失示例', alpha=0.7)
        ax.plot(x, 8 * np.exp(-x/15) + np.random.normal(0, 0.3, 100), label='验证损失示例', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('示例训练曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# 运行命令: streamlit run streamlit_app.py