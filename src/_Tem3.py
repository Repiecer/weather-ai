# dash_app.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import base64
import io

# 初始化Dash应用
app = dash.Dash(__name__, title="房价预测模型调试器")
server = app.server

# 布局
app.layout = html.Div([
    html.H1("🏠 房价预测模型调试器", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.H3("⚙️ 模型参数"),
            
            html.Label("学习率"),
            dcc.Slider(
                id='lr-slider',
                min=1e-5,
                max=0.01,
                step=1e-4,
                value=0.001,
                marks={i/1000: f'{i/1000:.3f}' for i in range(1, 11)},
                tooltip={"placement": "bottom"}
            ),
            
            html.Label("批次大小", style={'marginTop': '20px'}),
            dcc.Dropdown(
                id='batch-dropdown',
                options=[
                    {'label': '16', 'value': 16},
                    {'label': '32', 'value': 32},
                    {'label': '64', 'value': 64},
                    {'label': '128', 'value': 128},
                    {'label': '256', 'value': 256}
                ],
                value=64,
                clearable=False
            ),
            
            html.Label("训练轮数", style={'marginTop': '20px'}),
            dcc.Slider(
                id='epoch-slider',
                min=10,
                max=200,
                step=10,
                value=50,
                marks={i: str(i) for i in range(10, 201, 20)},
                tooltip={"placement": "bottom"}
            ),
            
            html.Label("Dropout率", style={'marginTop': '20px'}),
            dcc.Slider(
                id='dropout-slider',
                min=0,
                max=0.5,
                step=0.05,
                value=0.1,
                marks={i/10: f'{i/10:.1f}' for i in range(0, 6)},
                tooltip={"placement": "bottom"}
            ),
            
            html.Label("隐藏层结构", style={'marginTop': '20px'}),
            dcc.Input(
                id='hidden-input',
                type='text',
                value='128,64,32',
                style={'width': '100%'}
            ),
            
            html.Label("损失函数", style={'marginTop': '20px'}),
            dcc.Dropdown(
                id='loss-dropdown',
                options=[
                    {'label': 'MSE', 'value': 'mse'},
                    {'label': 'Huber', 'value': 'huber'},
                    {'label': 'MAE', 'value': 'mae'}
                ],
                value='huber',
                clearable=False
            ),
            
            html.Label("优化器", style={'marginTop': '20px'}),
            dcc.Dropdown(
                id='optimizer-dropdown',
                options=[
                    {'label': 'Adam', 'value': 'adam'},
                    {'label': 'AdamW', 'value': 'adamw'},
                    {'label': 'SGD', 'value': 'sgd'}
                ],
                value='adamw',
                clearable=False
            ),
            
            html.Button(
                '🚀 开始训练',
                id='train-button',
                n_clicks=0,
                style={
                    'marginTop': '30px',
                    'width': '100%',
                    'height': '50px',
                    'fontSize': '18px'
                }
            ),
            
            html.Div(id='loading-output', style={'marginTop': '20px'})
            
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'}),
        
        html.Div([
            html.H3("📊 训练结果"),
            
            dcc.Graph(id='loss-graph'),
            
            html.Div(id='results-div', style={'marginTop': '20px'})
            
        ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'})
    ])
])

@app.callback(
    [Output('loss-graph', 'figure'),
     Output('results-div', 'children'),
     Output('loading-output', 'children')],
    [Input('train-button', 'n_clicks')],
    [State('lr-slider', 'value'),
     State('batch-dropdown', 'value'),
     State('epoch-slider', 'value'),
     State('dropout-slider', 'value'),
     State('hidden-input', 'value'),
     State('loss-dropdown', 'value'),
     State('optimizer-dropdown', 'value')]
)
def update_output(n_clicks, lr, batch_size, epochs, dropout, hidden_str, loss_fn, optimizer):
    if n_clicks == 0:
        # 初始状态显示空图表
        fig = go.Figure()
        fig.update_layout(
            title="点击'开始训练'按钮开始",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=500
        )
        return fig, html.Div("等待训练..."), ""
    
    try:
        # 解析隐藏层
        hidden_layers = [int(x.strip()) for x in hidden_str.split(',') if x.strip()]
        
        # 加载数据
        file_path = '../datasets/housingPrice/data.xls'
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
        
        model = Net(X_train.shape[1], hidden_layers, dropout)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # 训练
        if loss_fn == "mse":
            criterion = torch.nn.MSELoss()
        elif loss_fn == "huber":
            criterion = torch.nn.HuberLoss(delta=1.0)
        else:
            criterion = torch.nn.L1Loss()
        
        if optimizer == "adam":
            optimizer_obj = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer == "adamw":
            optimizer_obj = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            optimizer_obj = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0
            for i in range(0, len(X_train_scaled), batch_size):
                batch_x = torch.FloatTensor(X_train_scaled[i:i+batch_size]).to(device)
                batch_y = torch.FloatTensor(y_train_scaled[i:i+batch_size]).to(device)
                
                optimizer_obj.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer_obj.step()
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
        
        # 创建图表
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(train_losses) + 1)),
            y=train_losses,
            mode='lines',
            name='训练损失',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(1, len(val_losses) + 1)),
            y=val_losses,
            mode='lines',
            name='验证损失',
            line=dict(color='orange', width=2)
        ))
        
        fig.update_layout(
            title='训练过程',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            height=500,
            hovermode='x unified'
        )
        
        # 创建结果面板
        results = html.Div([
            html.H4("📈 训练结果"),
            html.Table([
                html.Tr([
                    html.Td("最终训练损失:"),
                    html.Td(f"{train_losses[-1]:.4f}")
                ]),
                html.Tr([
                    html.Td("最终验证损失:"),
                    html.Td(f"{val_losses[-1]:.4f}")
                ]),
                html.Tr([
                    html.Td("最佳验证损失:"),
                    html.Td(f"{min(val_losses):.4f} (Epoch {val_losses.index(min(val_losses)) + 1})")
                ]),
                html.Tr([
                    html.Td("过拟合程度:"),
                    html.Td(f"{(train_losses[-1] - val_losses[-1]) / val_losses[-1] * 100:.2f}%")
                ])
            ], style={'width': '100%', 'marginTop': '10px'})
        ])
        
        return fig, results, html.Div("✅ 训练完成！", style={'color': 'green'})
        
    except Exception as e:
        error_div = html.Div([
            html.H4("❌ 错误"),
            html.P(str(e))
        ], style={'color': 'red'})
        
        return go.Figure(), error_div, ""

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)