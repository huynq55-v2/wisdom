import os

input_file = "wisdom_models/replay_buffer.csv"
output_dir = "wisdom_models/buffers/v0"

os.makedirs(output_dir, exist_ok=True)

with open(input_file, 'r') as f:
    lines = f.readlines()

shard_size = 10000
for i in range(0, len(lines), shard_size):
    shard_lines = lines[i:i+shard_size]
    out_path = f"{output_dir}/shard_legacy_{i//shard_size}.csv"
    with open(out_path, 'w') as out:
        out.writelines(shard_lines)

print(f"✅ Đã chia {len(lines)} FENs thành {len(lines)//shard_size + 1} file shards nằm trong {output_dir}!")
