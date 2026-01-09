# 动量法的直观理解
def momentum_visualization():
    """动量法的物理类比：小球下山"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 模拟损失函数地形
    x = np.linspace(-2, 2, 100)
    y = x**2 + 0.5*np.sin(10*x)  # 有局部震荡的地形
    
    plt.figure(figsize=(12, 5))
    
    # SGD（无动量）
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'b-', label='损失函数')
    
    # 模拟SGD路径（容易卡在局部震荡）
    sgd_path = []
    pos = 1.8
    for _ in range(20):
        grad = 2*pos + 2*np.cos(10*pos)  # 梯度
        pos -= 0.1 * grad  # SGD更新
        sgd_path.append(pos)
    
    plt.plot(sgd_path, [p**2 + 0.5*np.sin(10*p) for p in sgd_path], 
             'ro-', label='SGD路径')
    plt.title('SGD（无动量）: 容易震荡')
    plt.xlabel('参数')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Momentum（有动量）
    plt.subplot(1, 2, 2)
    plt.plot(x, y, 'b-', label='损失函数')
    
    # 模拟Momentum路径
    momentum_path = []
    pos = 1.8
    velocity = 0
    momentum = 0.9
    for _ in range(20):
        grad = 2*pos + 2*np.cos(10*pos)
        velocity = momentum * velocity - 0.1 * grad
        pos += velocity
        momentum_path.append(pos)
    
    plt.plot(momentum_path, [p**2 + 0.5*np.sin(10*p) for p in momentum_path], 
             'go-', label='Momentum路径')
    plt.title('Momentum: 平滑收敛')
    plt.xlabel('参数')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 运行可视化
momentum_visualization()