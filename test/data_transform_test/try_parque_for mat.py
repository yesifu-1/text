import pandas as pd

# 创建一个示例数据框（DataFrame）
data = {
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "score": [88.5, 92.0, 79.5]
}
df = pd.DataFrame(data)

# 🔽 保存为 Parquet 格式（默认使用 pyarrow）
df.to_parquet("example.parquet", index=False)
breakpoint()
# ✅ 读取 Parquet 文件
df_loaded = pd.read_parquet("example.parquet")

# 打印验证
print("原始数据：")
print(df)

print("\n读取后的数据：")
print(df_loaded)
