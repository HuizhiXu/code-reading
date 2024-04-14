# pip install -qU bitsandbytes datasets peft trl accelerate

import os
import torch
import re
import json
import gdown
from datasets import Dataset
import pandas as pd
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, GenerationConfig
from tqdm.auto import tqdm
from trl import DPOTrainer

# Download the dataset and load it
# git clone https://github.com/Baiiiiiiiiii/GenAI_hw6_dataset.git
with open("/content/GenAI_hw6_dataset/labelled_data.json", 'r') as jsonfile:
    full_data = json.load(jsonfile)

with open("/content/GenAI_hw6_dataset/test_prompt.json", 'r') as jsonfile:
    test_data = json.load(jsonfile)


# 使用AutoModelForCausalLM这个类来加载一个预训练的因果语言模型（Causal Language Model）
"""
quantization_config# 这个参数指定了一个量化配置对象，它定义了如何对模型进行量化以减少其大小和加速推理。
量化是一种减少模型权重精度的技术，可以节省内存并提高推理速度，通常以牺牲一些模型精度为代价。
load_in_4bit=True: 模型的权重将被量化为4位的表示。量化是减少模型大小和加速推理的一种技术。在4位量化中，权重和激活值被表示为4位的数字，这极大地减少了模型的内存占用。然而，这种低精度表示可能会导致精度损失，因此需要仔细调整以保持模型性能。
bnb_4bit_compute_dtype=torch.bfloat16: 这是在4位量化模式下用于计算的数据类型，torch.bfloat16是一种16位宽的浮点数据类型，它提供了比32位浮点数更高的性能。
bnb_4bit_use_double_quant=True: 这表示在量化过程中将使用双精度量化。
bnb_4bit_quant_type='nf4': 这是指定的量化类型，'nf4'代表一种特定的量化方案。

4位精度 (load_in_4bit=True):
这意味着模型的权重将被量化为4位的表示。量化是减少模型大小和加速推理的一种技术。在4位量化中，权重和激活值被表示为4位的数字，这极大地减少了模型的内存占用。然而，这种低精度表示可能会导致精度损失，因此需要仔细调整以保持模型性能。

BFloat16 (bnb_4bit_compute_dtype=torch.bfloat16):
BFloat16（或BF16）是一种16位宽的浮点数据类型。它具有与32位浮点数（FP32）相同的指数位宽度，但较少的尾数位。这使得BFloat16在表示非常大或非常小的数字时比FP32更有效，同时在许多深度学习应用中提供了足够的精度。BFloat16通常用于混合精度训练，其中模型的一部分使用较低的精度（如4位或8位）以节省内存和加速计算，而另一部分使用BFloat16以保持精度。

双精度量化 (bnb_4bit_use_double_quant=True):
双精度量化是一种量化策略，它使用两个不同的量化级别来表示权重。这通常意味着模型的一部分权重使用较低的精度（如4位），而另一部分权重使用较高的精度（如BFloat16）。这种方法旨在平衡模型大小和计算效率与模型精度之间的关系。

NF4量化类型 (bnb_4bit_quant_type='nf4'):
NF4是一种特定的量化方案，它可能是指一种非均匀量化（non-uniform quantization）或非线性量化方法。这种量化方法与传统的均匀量化（每个级别之间的间隔相同）不同，它允许在不同的量化级别之间有不同的间隔，以便更好地保持模型的精度。


"""
model = AutoModelForCausalLM.from_pretrained(
    'MediaTek-Research/Breeze-7B-Instruct-v0_1', # 预训练模型的标识符，指定了要从Hugging Face模型库中加载的模型。
    device_map='auto', #指示Transformers库自动决定模型应该加载到哪个设备上,有GPU就加载到GPU上，没有就加载到CPU上
    trust_remote_code=True, # 设置为True意味着你信任远程加载的代码，并且允许执行可能的远程代码。这通常是安全的，因为Transformers库会确保代码的安全性，但是如果你想要更加谨慎，可以将其设置为False。
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )  
)

# 
# 使用Hugging Face的Transformers库中的预训练模型和分词器（tokenizer）来生成文本。
# 这个过程是一个典型的基于预训练语言模型的文本生成任务，其中模型根据给定的提示生成响应。这种技术可以用于聊天机器人、自动问答系统等多种应用场景。
"""
1. 加载预训练分词器（tokenizer）:
2. 设置分词器的填充参数：padding_side = "right"意味着在文本的右侧进行填充。pad_token是用于填充的特殊token，这里被设置为分词器的结束符（eos_token），这通常用于指示分词器生成的序列已经结束。
3. 定义数据处理函数data_formulate(data)：data_formulate函数接收一个包含提示信息的字典data，然后构建一个包含系统消息和用户消息的消息列表。
        apply_chat_template函数使用分词器的聊天模板功能来处理这些消息，并生成一个用于模型生成的提示字符串。
4. 处理测试数据并生成响应：
    - 使用data_formulate函数生成处理后的提示，然后使用分词器将其转换为模型可以理解的格式，并指定返回的张量类型为PyTorch张量（return_tensors="pt"）。
    - GenerationConfig对象用于配置文本生成的参数。在这个例子中，do_sample=False意味着模型将不会随机采样token，
    而是使用最可能的token；max_new_tokens = 200限制了生成的token数量；pad_token_id是用于填充的token的ID。
    - model.generate函数接收输入数据和生成配置，生成文本响应。生成的输出是一个token ID的列表，
        然后使用tokenizer.batch_decode将其转换为可读的文本，并去除特殊token。最后，输出被添加到original_model_response列表中，并打印出来。

"""
tokenizer = AutoTokenizer.from_pretrained('MediaTek-Research/Breeze-7B-Instruct-v0_1')
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token


def data_formulate(data):
    messages = [
        {"role": "system", "content": '回覆請少於20字'},
        {"role": "user", "content": data['prompt']},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

original_model_response = []
for data in tqdm(test_data):
    id = data['id']
    print(f'Question {id}:\n'+data['prompt'])
    inputs = tokenizer(data_formulate(data), return_tensors="pt").to('cuda')
    generation_config=GenerationConfig(
            do_sample=False,
            max_new_tokens = 200,
            pad_token_id = tokenizer.pad_token_id
    )
    output = model.generate(**inputs, generation_config=generation_config)
    output = tokenizer.batch_decode(output, skip_special_tokens=True)[0].split('[/INST] ')[1]
    original_model_response.append(output)
    print('Response from original model:\n'+output+'\n')



# 设置参数 
num_epoch = 2
data_size = 50
support_ratio = 0



training_data = full_data[:data_size]

# 这里计算了支持数据集的大小，它是基于总训练数据样本数量data_size和支持比例support_ratio。
# 支持数据集可能是指用于训练模型的正面样本集合，而剩余的样本用于训练模型识别非期望的行为或结果。

support_data_size = int(data_size * support_ratio)

# Prepare the data for the training dataset
"""
prompt_list：使用data_formulate函数处理每个训练样本，生成对应的提示。
chosen_list：结合了支持数据集中的'support'样本和反对数据集中的'oppose'样本，这些可能是被模型视为“正确”或“首选”的样本。
rejected_list：结合了支持数据集中的'oppose'样本和反对数据集中的'support'样本，这些可能是被模型视为“错误”或“非首选”的样本。
position_list：定义了每个样本的类别标签，前support_data_size个样本标记为'support'，其余样本标记为'oppose'。
"""
prompt_list = [data_formulate(data) for data in training_data]
chosen_list = [data['support'] for data in training_data[:support_data_size]] + [data['oppose'] for data in training_data[support_data_size:]]
rejected_list = [data['oppose'] for data in training_data[:support_data_size]] + [data['support'] for data in training_data[support_data_size:]]
position_list = ['support' for _ in range(support_data_size)] + ['oppose' for _ in range(data_size - support_data_size)]

# Create the training dataset
train_dataset = Dataset.from_dict({'prompt': prompt_list, 'position': position_list, 'chosen': chosen_list, 'rejected': rejected_list})
pd.DataFrame(train_dataset).rename(columns={"chosen": "preferred", "rejected": "non-preferred"}) # 最后，将train_dataset转换为Pandas的DataFrame，并重命名列名，将'chosen'改为'preferred'，将'rejected'改为'non-preferred'。这可能是为了使数据集的列名更符合模型的预期输入格式或后续分析的需要。


"""
使用Hugging Face的Transformers库中的TrainingArguments类来配置训练参数。
output_dir='./'：
指定训练过程中模型检查点、日志和其他输出文件的保存目录。这里设置为当前目录（.）。

per_device_train_batch_size=1：
设置每个设备（如GPU或CPU）上的批量大小。这里设置为1，意味着每次训练迭代只处理一个样本。这通常用于减少内存消耗，特别是在处理大型模型时。

num_train_epochs=num_epoch：
设置训练过程中数据集将被遍历的次数。这里的num_epoch变量应该在代码的其他部分定义，表示训练的轮数。

gradient_accumulation_steps=8：
设置梯度累积的步数。当批量大小较小或使用梯度累积技术时，这个参数会很有用。在这里，每8个训练步骤才会执行一次梯度更新。

gradient_checkpointing=False：
是否启用梯度检查点技术来减少内存使用。这里设置为False，意味着不使用梯度检查点。

learning_rate=2e-4：
设置学习率，这是优化算法中的一个重要参数，控制着权重更新的幅度。这里设置为2e-4，即0.0002。

optim="paged_adamw_8bit"：
设置优化器。这里使用的是paged_adamw_8bit，这可能是一个自定义的优化器或特定于某个库的优化器。它可能结合了AdamW优化器的特性和8位量化技术以提高训练效率。

logging_steps=1：
设置日志记录的频率。这里设置为1，意味着每次训练步骤都会记录日志。

warmup_ratio=0.1：
设置学习率预热的比例。预热是学习率调度的一种技术，它在训练开始时以较小的学习率开始，然后逐渐增加到设定的学习率。这里的0.1意味着预热阶段的学习率是最大学习率的10%。

report_to = 'none'：
设置训练过程中的性能报告目标。这里设置为'none'，意味着不向任何外部系统报告训练进度。

"""
training_args = TrainingArguments(
    output_dir='./',
    per_device_train_batch_size=1,
    num_train_epochs=num_epoch,
    gradient_accumulation_steps=8,
    gradient_checkpointing=False,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    logging_steps = 1,
    warmup_ratio = 0.1,
    report_to = 'none'
)
"""
LoraConfig类被用来创建一个配置对象，这个对象包含了用于LoRA（Low-Rank Adaptation）技术的参数。LoRA是一种模型微调技术，它通过在预训练模型的权重上应用低秩矩阵来调整和优化模型，而不是直接修改原始的预训练权重。这种方法可以减少微调过程中需要学习的参数数量，从而节省内存和计算资源。以下是对LoraConfig中各个参数的详细解释：

lora_alpha=16：
这个参数指定了LoRA操作中使用的缩放因子。在LoRA中，原始权重矩阵被分解为两个较小的矩阵的乘积，其中一个矩阵的权重被缩放。lora_alpha控制这个缩放的程度。较高的值意味着更大的变化，较低的值则意味着更保守的权重调整。

lora_dropout=0.1：
这个参数设置了在LoRA操作中使用的dropout比率。Dropout是一种正则化技术，它可以防止模型过拟合，通过在训练过程中随机丢弃一些权重来实现。在这里，lora_dropout参数指定了在LoRA过程中要丢弃的权重比例。

r=64：
这个参数定义了LoRA操作中使用的秩（rank）。秩是指在权重矩阵分解过程中，两个较小矩阵的秩。较低的秩意味着模型的复杂度较低，而较高的秩则允许模型具有更高的复杂度。秩的选择取决于特定任务的需求和可用的计算资源。

bias="none"：
这个参数指定了是否在LoRA操作中使用偏置项。在这里，bias="none"表示不使用偏置项，这可能会减少模型的参数数量并简化模型结构。

task_type="CAUSAL_LM"：
这个参数定义了当前任务的类型。在这个例子中，"CAUSAL_LM"表示任务是因果语言模型（Causal Language Model），这类模型通常用于生成文本或预测下一个词。因果语言模型在处理序列数据时只考虑之前的元素，而不是整个序列。

"""

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",

)

"""

DPOTrainer 是一种基于直接偏好优化（Direct Preference Optimization, DPO）的机器学习方法，它专注于根据人类偏好数据直接优化语言模型，而不是依赖于传统的强化学习（Reinforcement Learning, RL）方法中的奖励模型拟合和策略优化步骤。

在传统的基于人类反馈的强化学习（RLHF）流程中，通常包括以下几个步骤：监督学习（Supervised Learning）、奖励模型训练（Reward Modeling）、强化学习策略优化（RL Optimization）等。
而DPO方法的关键在于直接利用人类偏好数据来优化语言模型，从而避免了RLHF流程中奖励模型拟合和RL优化的复杂性和不稳定性。

DPOTrainer 通过比较人类标注的偏好样本（例如，一对回答中的优选回答和非优选回答）来训练模型，使模型更倾向于生成符合人类偏好的输出。
这种方法简化了偏好学习的过程，因为它不需要显式地训练一个奖励模型，也不需要使用复杂的强化学习算法来优化策略。

DPOTrainer 通常需要以下组件和步骤来执行训练：

基础模型（Base Model）：一个已经过监督学习调优（SFT）的模型，作为DPO优化的基础。
参考模型（Reference Model）：通常与基础模型架构相同，用于计算偏好和非偏好样本的隐式奖励。
偏好数据集（Preference Dataset）：包含人类偏好标注的数据，用于训练模型以优化偏好。
DPO损失函数（DPO Loss Function）：结合了基础模型和参考模型生成的概率，以及一个温度参数（beta），用于指导训练过程。
训练参数（Training Parameters）：如学习率、批量大小等，用于配置训练循环的参数。



DPOTrainer是一个用于训练和微调深度学习模型的类。它整合了模型训练所需的各种组件和参数。以下是对DPOTrainer构造函数中各个参数的详细解释：

model：
这是要训练或微调的模型对象。它应该是一个已经加载并配置好的模型，例如使用Hugging Face的Transformers库中的AutoModelForCausalLM或其他类似类的实例。

args=training_args：
这是一个TrainingArguments对象，包含了训练过程中的所有必要参数，例如学习率、批量大小、训练轮数等。这些参数通常用于配置训练循环和优化器。

beta=0.1：
这个参数可能是用于正则化或权重衰减的系数。在许多优化算法中，beta用于控制L2正则化或Adam优化器的权重衰减，以防止过拟合。在这里，0.1表示应用了相对较轻的正则化。

train_dataset=train_dataset：
这是训练数据集对象，它包含了用于训练模型的数据。数据集对象应该实现了必要的方法，例如__len__()来返回数据集的大小，以及__getitem__()来获取单个样本。

tokenizer=tokenizer：
这是分词器对象，用于将原始文本数据转换为模型可以理解的格式。分词器通常负责文本的分词、编码、转换为数字ID等任务。

peft_config=peft_config：
这是一个PeftConfig对象（可能是LoRAConfig的一个变体或相关类），包含了与LoRA（Low-Rank Adaptation）技术相关的配置参数。这些参数用于指导模型在微调过程中如何应用LoRA技术来优化权重。

"""

dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)


dpo_trainer.train()


# Generate responses using the trained model and compare with the original model
"""
 这段代码展示了如何使用训练好的模型在测试数据集上生成响应，并与原始模型的响应进行比较的过程。具体步骤如下：

1. **生成训练模型的响应**：
   - 通过遍历测试数据集`test_data`，使用`tqdm`库来显示进度条。
   - 对于每个测试数据项，首先提取或生成其ID和提示（prompt）。
   - 使用分词器`tokenizer`将处理后的提示转换为模型可以理解的格式，并通过`.to('cuda')`将其发送到GPU进行加速。
   - 设置`GenerationConfig`，指定不使用采样（`do_sample=False`），最大生成的新token数量为200，以及填充token的ID。
   - 使用模型的`generate`方法根据输入提示生成文本响应。
   - 将生成的输出通过分词器的`batch_decode`方法转换回文本，并去除特殊token。
   - 将生成的响应添加到列表`trained_model_response`中，并打印出来。

2. **比较原始模型和训练模型的响应**：
   - 打印出训练过程中使用的参数，如训练轮数`num_epoch`、数据集大小`data_size`和支持比例`support_ratio`。
   - 再次遍历测试数据集，对于每个数据项：
     - 提取原始模型的响应（`ref_output`）和训练模型的响应（`output`）。
     - 打印出测试数据的ID、提示、原始模型的响应和训练模型的响应。
   - 创建一个字典`model_response`，包含每个测试数据的ID、提示、原始模型的响应和训练模型的响应，然后将这个字典添加到列表中。


"""
trained_model_response = []
for data in tqdm(test_data):
    id = data['id']
    print(f'Question {id}:\n'+data['prompt'])
    inputs = tokenizer(data_formulate(data), return_tensors="pt").to('cuda')
    generation_config=GenerationConfig(
            do_sample=False,
            max_new_tokens = 200,
            pad_token_id = tokenizer.pad_token_id
    )
    output = model.generate(**inputs, generation_config=generation_config)
    output = tokenizer.batch_decode(output, skip_special_tokens=True)[0].split('[/INST] ')[1]
    trained_model_response.append(output)
    print('Response from trained model:\n'+output+'\n')


model_response = []
print(f'num_epoch: {num_epoch}\ndata_size: {data_size}\nsupport_ratio: {support_ratio}')
print()
for data in test_data:
    id = data['id']
    ref_output = original_model_response[id-1]
    output = trained_model_response[id-1]
    print(f'Question {id}:\n'+data['prompt'])
    print('Response from original model:\n'+ref_output)
    print('Response from trained model:\n'+output)
    print()
    model_response.append({'id':data['id'], 'prompt':data['prompt'], 'response_from_original_model':ref_output, 'response_from_trained_model':output})