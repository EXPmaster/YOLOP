### 整体结构

├─lib
│  ├─config	# 不太用管，用于管理yaml文件
│  ├─core	  # 包括inference 推理代码， function(train和validate函数)，evaluate代码和loss损失函数代码
│  ├─dataset # 包括父数据集和子数据集
│  ├─models 包括backbone和heads, factory.py用于组合backbone和heads
│  └─utils	# 各种乱七八糟的，其实也可以再新建文件夹
├─tools  存放train.py和test.py 
└─visualization	plot.py用于读取结果的json文件，并绘制结果在图片上



### 补充说明

net1.yaml里面的内容，可以自行充实添加

AUTO_RESUME: true # 是否自动从上次断掉的地方接着开始训练

CUDNN:      # 关于CUDA的三个配置，这个我也不是很了解

 BENCHMARK: true

 DETERMINISTIC: false

 ENABLED: true

GPUS: (0,1,2,3)  # 使用的gpu

OUTPUT_DIR: 'output' # 保存模型文件

LOG_DIR: 'log' # 保存log文件

WORKERS: 24   # 用几个worker

PRINT_FREQ: 100  # 每100次print一次结果



DATASET:

 COLOR_RGB: False  # 是否使用RGB的图片

 DATASET:      # 使用的dataset的名字比如bdd，就是你在dataset文件夹里面用的那个名字

 DATA_FORMAT: jpg  # 图片格式，也许没用，看你需不需要，可以删掉

 FLIP: true     # 是否采用左右翻转

 ROOT: 'data/aichallenger/'  # 数据集路径

 ROT_FACTOR: 30   # 旋转角度

 SCALE_FACTOR: 0.25 # 缩放大小

 TEST_SET: valid   # /root/test_set

 TRAIN_SET: train

 SELECT_DATA: False

MODEL:

 NAME: 'resnet'

 PRETRAINED: ''

 IMAGE_SIZE:

 \- 256

 \- 256

 HEADS_NAME: ['']

LOSS:

 LOSS_NAME: ''  # 你使用的loss的名称

TRAIN:

 BATCH_SIZE_PER_GPU: 32   # 每张卡上的Batchsize

 SHUFFLE: true        # 是否对dataset进行shuffle

 BEGIN_EPOCH: 0

 END_EPOCH: 140       # 多少个epoch结束

 VAL_FREQ: 10        # 多少个epoch做一次validate

 OPTIMIZER: 'adam'      # 使用什么optimizer

 LR: 0.001

 LR_FACTOR: 0.1       # 使用lr_scheduler时的参数，到了固定step的时候对现有LR乘以LR_FACTOR

 LR_STEP:          # 使用lr_scheduler时的参数，到了这个step对LR乘以参数

 \- 90

 \- 120

 WD: 0      # Weight decay

 GAMMA1: 0.99  # 剩下四个好像都是sgd里面用的，如果用adam不用管它们

 GAMMA2: 0.0

 MOMENTUM: 0.9

 NESTEROV: false

TEST:

 BATCH_SIZE_PER_GPU: 32

 MODEL_FILE: ''   # 模型文件