import matplotlib.pyplot as plt

# 模型评估结果数据
data = [
    {
        'model': 'Resnet-18',
        'rank1': 0.12,
        'rank5': 0.65,
        'rank10': 1.22,
        'mAP': 0.19,
        'mINP': 0.09,
        'metric': 0.15
    },
    {
        'model': 'VehicleID',
        'rank1': 0.97,
        'rank5': 2.47,
        'rank10': 3.71,
        'mAP': 0.60,
        'mINP': 0.13,
        'metric': 0.78
    },
    {
        'model': 'People-Reid(without training)',
        'rank1': 0.93,
        'rank5': 2.55,
        'rank10': 3.80,
        'mAP': 0.62,
        'mINP': 0.13,
        'metric': 0.77
    },
    {
        'model': 'People-Reid(trained)',
        'rank1': 67.95,
        'rank5': 81.73,
        'rank10': 88.68,
        'mAP': 66.71,
        'mINP': 51.75,
        'metric': 67.33
    },
    {
        'model': 'Bike-Reid(Trained)',
        'rank1': 56.25,
        'rank5': 71.43,
        'rank10': 79.53,
        'mAP': 52.79,
        'mINP': 34.39,
        'metric': 54.52
    },
]

# 提取数据
x = ['rank1', 'rank5', 'rank10', 'mAP', 'mINP', 'metric']
models = [d['model'] for d in data]
y_values = {
    model: [d[key] for key in x] for model, d in zip(models, data)
}

# 绘制折线图
for model, values in y_values.items():
    plt.plot(x, values, label=model)

# 设置图例和标题
plt.legend()
plt.title('Model Evaluation')

# 显示折线图
plt.tight_layout()
plt.show()
