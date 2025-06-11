# 0. experiments
RANDOM_SEED = 42
# 1. node
EDGE_NODE_NUM = 15

node_mem_max = 180
node_mem_min = 80

node_disk_max = 600
node_disk_min = 450

node_cpu_freq_max = 650
node_cpu_freq_min = 300

node_band_max = 6
node_band_min = 2
# 2. image
IMAGE_NUM = 50
image_size_min = 3
image_size_max = 15

# 3. task
# TASK_NUM = 100000

task_mem_max = 10

task_cpu_freq_max = 250
task_cpu_freq_min = 150
task_size_max = 10

# 4. pos
max_x = 100
max_y = 100

# 5. other
vit_dim = EDGE_NODE_NUM

# 6. Energy consumption ratio
max_E = 10
min_E = 5

#8. Transmission energy consumption
max_t = 0.0001995

#7. Parameters of energy consumption and ddl in reward
e1 = 1
e2 = 1

epoch = 400

max_tasks = 20
min_tasks = 5

epoch_imitation = 20