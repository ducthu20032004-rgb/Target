import pandas as pd
import matplotlib.pyplot as plt

# Từ điển ánh xạ label → màu
colors = {
    'Task 0': 'blue',
    'Task 1': 'orange',
    'Task 2': 'red',
    'Task 3': 'green',
    'Task 4': 'purple',
    # 'FFA-LoRA': 'orange',
    # 'FedSA-LoRA': 'red',
    # 'FLoRA-CA': 'black',
}

# Custom labels bạn muốn vẽ
custom_labels = ['Task 0', 'Task 1', 'Task 2', 'Task 3', 'Task 4']

# Dữ liệu
df = pd.read_csv('Task_all_acc.csv')
x = df.iloc[:, 0]
max_columns = [col for col in df.columns if col.endswith('_acc')]

# Map label to columns theo thứ tự custom_labels
label_to_column = dict(zip(custom_labels, max_columns))

# Sắp xếp các label theo thứ tự trong `colors`
sorted_labels = [label for label in colors if label in custom_labels]

# Nếu có label không trong `colors`, thêm vào cuối
other_labels = [label for label in custom_labels if label not in colors]
final_labels = sorted_labels + other_labels

# Cấu hình smoothing
ema_span = 30
std_window = 3
plt.rcParams.update({'font.size': 18})

# Vẽ hình
plt.figure(figsize=(8, 6))
for label in final_labels:
    if label not in label_to_column:
        print(f"⚠️ Label '{label}' không khớp với bất kỳ cột dữ liệu nào.")
        continue

    col = label_to_column[label]
    color = colors.get(label, 'black')  # fallback nếu label không có màu

    # ema = df[col].ewm(span=ema_span, adjust=False).mean()
    # std = df[col].rolling(window=std_window, min_periods=1).std()

    linestyle = '--' if label == 'STAMP' else '-'  # Dùng nét đứt cho STAMP
    plt.plot(x,df[col], marker='x', label=label, linewidth=2.5, color=color, linestyle=linestyle)
    # plt.fill_between(x, ema - std, ema + std, alpha=0.2, color=color)

# Giao diện
plt.xlabel('Task Steps')
plt.ylabel('Acc')
plt.title('Task all Fixed 5 heads Accuracy')
plt.legend()
plt.grid(True)
plt.xlim(left=0, right=4)
plt.ylim(bottom=0, top=100)
plt.tight_layout()
plt.savefig("MNLI-grad.pdf", bbox_inches='tight')
plt.show()
