"""
简单的 Benchmark 评测模块
支持 C3 和 XCOPA 数据集的评测
"""
import json
import torch
import torch.nn.functional as F


def eval_multiple_choice(model, tokenizer, context, choices, label_idx, max_length=512):
    """
    多选题评测：计算每个选项的困惑度，选择困惑度最低的
    
    Args:
        model: 语言模型
        tokenizer: tokenizer
        context: 问题上下文（字符串）
        choices: 选项列表（字符串列表）
        label_idx: 正确答案的索引
        max_length: 最大序列长度
    
    Returns:
        1 表示预测正确,0 表示预测错误
    """
    # 1. 初始化空列表：存储每个选项对应的平均损失（损失越低=困惑度越低=模型越认可该选项）
    losses = []
    
    # 2. 遍历每个候选选项，逐个计算损失
    for choice in choices:
        # 3. 拼接完整文本：问题上下文 + 当前选项（如“外面下雨了没带伞” + “待在家里”）
        full_text = context + choice
        
        # 4. 文本编码：将拼接后的文本转为模型可识别的token_ids
        #    return_tensors="pt"：返回PyTorch张量
        #    max_length/truncation：限制最大长度并截断超长文本，避免显存溢出
        #    .to(model.device)：将张量移到模型所在设备（GPU/CPU）
        inputs = tokenizer(
            full_text, 
            return_tensors="pt", 
            max_length=max_length,
            truncation=True
        ).to(model.device)
        # 5. 提取编码后的token_ids（模型输入的核心）
        input_ids = inputs.input_ids
        
        # 6. 单独编码上下文文本：目的是计算上下文的token长度，用于定位选项部分
        #    add_special_tokens=True：和完整文本编码保持一致（避免长度计算偏差）
        context_tokens = tokenizer(context, add_special_tokens=True).input_ids
        # 7. 得到上下文的token数量（如“外面下雨了没带伞”编码后有8个token）
        context_len = len(context_tokens)
        
        # 8. 前向传播计算logits（禁用梯度计算，节省显存+加速）
        with torch.no_grad():
            # 9. 模型前向推理：输入token_ids，输出每个位置的预测logits
            outputs = model(input_ids=input_ids)
            # 10. 提取logits（形状：[1, seq_len, vocab_size]，表示每个位置预测每个词的得分）
            logits = outputs.logits
        
        # 11. 计算损失：只关注选项部分的损失（核心逻辑）
        # 12. 移位logits：logits[:, :-1, :] 对应预测 input_ids[:, 1:]（因为logits[i]预测input_ids[i+1]）
        shift_logits = logits[..., :-1, :].contiguous()
        # 13. 移位标签：作为损失计算的真实标签
        shift_labels = input_ids[..., 1:].contiguous()
        
        # 14. 定义交叉熵损失函数：reduction='none' 保留每个位置的损失（不平均）
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        # 15. 计算每个token位置的损失（形状：[seq_len-1]）
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # 16. 计算选项部分在损失中的起始位置（关键！）
        #     原理：loss[i] 对应预测原始序列中 i+1 位置的token
        #     上下文占 0~context_len-1 位置 → 选项从 context_len 位置开始
        #     因此选项部分的损失起始位置是 context_len - 1
        choice_start = max(0, context_len - 1)
        
        # 17. 只计算选项部分的平均损失（模型对该选项的“认可程度”）
        if choice_start < len(loss):
            # 18. 取选项部分的所有损失，计算平均值并转为Python浮点数
            choice_loss = loss[choice_start:].mean().item()
        else:
            # 19. 兜底：如果上下文太长导致选项被截断，用整个序列的平均损失
            choice_loss = loss.mean().item()
        
        # 20. 将当前选项的损失加入列表
        losses.append(choice_loss)
    
    # 21. 选择损失最小的选项作为模型预测（损失越低=模型认为该选项越合理）
    pred_idx = losses.index(min(losses))
    # 22. 对比预测索引和正确答案索引：相等返回1（答对），否则返回0（答错）
    return 1 if pred_idx == label_idx else 0


def eval_c3(model, tokenizer, data_path):
    """
    评测 C3 数据集
    
    Args:
        model: 模型
        tokenizer: tokenizer
        data_path: C3 数据集路径（jsonl 格式）
    
    Returns:
        准确率（0-1之间的浮点数）
    """
    # 1. 初始化正确数计数器：统计模型答对的题目数量
    correct = 0
    # 2. 初始化总数计数器：统计有效评测的题目总数
    total = 0
    
    # 3. 以只读方式打开C3数据集文件（utf-8编码避免中文乱码）
    with open(data_path, 'r', encoding='utf-8') as f:
        # 4. 逐行读取JSONL格式的数据集（每行一个评测样本）
        for line in f:
            # 5. 去除每行首尾的空白字符（换行/空格），并解析为JSON字典
            data = json.loads(line.strip())
            
            # 6. 解析C3数据集的核心字段（C3固定格式）：
            #    context: 对话上下文（列表形式，如["A:你好","B:下雨了"]）
            #    question: 问题（字符串，如"接下来B会说什么？"）
            #    choice: 候选答案列表（如["带伞","跑步"]）
            #    answer: 正确答案（字符串，如"带伞"）
            # 7. 合并上下文列表为完整字符串（把分散的对话拼接成一段文本）
            context_text = ''.join(data['context'])  
            # 8. 提取问题文本
            question = data['question']
            # 9. 提取候选答案列表
            choices = data['choice']
            # 10. 提取正确答案文本
            answer = data['answer']
            
            # 11. 过滤无效样本：如果正确答案不在候选列表中，跳过该样本
            #     避免后续index查找报错，保证评测数据有效性
            if answer not in choices:
                continue
            # 12. 将正确答案文本转换为对应的索引（如["A","B","C"]中" B"对应索引1）
            #     方便后续和模型预测的索引对比
            label_idx = choices.index(answer)
            
            # 13. 构建模型输入的完整上下文：拼接对话上下文+问题，形成完整的推理背景
            full_context = context_text + question
            
            # 14. 调用多选推理核心函数，判断模型是否答对这道题
            #     输入：模型、tokenizer、完整上下文、候选选项、正确答案索引
            #     输出：1（答对）/0（答错）
            result = eval_multiple_choice(model, tokenizer, full_context, choices, label_idx)
            # 15. 累计答对的数量
            correct += result
            # 16. 累计有效样本总数
            total += 1
    
    # 17. 计算准确率：正确数/总数（避免除以0，总数为0时准确率记0）
    accuracy = correct / total if total > 0 else 0
    # 18. 返回最终准确率（0-1之间，如0.85代表85%正确率）
    return accuracy


def eval_xcopa(model, tokenizer, data_path):
    """
    评测 XCOPA 数据集
    
    Args:
        model: 模型
        tokenizer: tokenizer
        data_path: XCOPA 数据集路径（jsonl 格式）
    
    Returns:
        准确率（0-1之间的浮点数）
    """
    correct = 0
    total = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # XCOPA 数据格式：premise (str), choice1 (str), choice2 (str), question (str), label (int)
            premise = data['premise']
            choices = [data['choice1'], data['choice2']]
            label_idx = data['label']
            question_type = data['question']  # 'cause' 或 'effect'
            
            # 构建上下文（根据问题类型调整提示，使用更明确的格式）
            if question_type == 'cause':
                context = f"{premise}这是因为："
            else:  # effect
                context = f"{premise}所以："
            
            # 评测
            result = eval_multiple_choice(model, tokenizer, context, choices, label_idx)
            correct += result
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


def run_benchmark(model, tokenizer, c3_path, xcopa_path):
    """
    运行所有 benchmark 评测
    
    Args:
        model: 模型（会自动解包 DDP）
        tokenizer: tokenizer
        c3_path: C3 数据集路径
        xcopa_path: XCOPA 数据集路径
    
    Returns:
        包含所有评测结果的字典
    """
    # 1. 初始化空字典，用于存储最终的评测结果（C3和XCOPA的准确率）
    results = {}
    
    # 2. 导入分布式数据并行（DDP）模块，用于判断模型是否被DDP包装
    from torch.nn.parallel import DistributedDataParallel
    # 3. 解包模型：如果模型是DDP训练的（多卡），取原始模型；否则直接用原模型
    #    原因：DDP包装的模型不能直接用于评测，必须解包到原始模型
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    # 4. 进一步解包：如果模型被torch.compile编译过，取编译前的原始模型（_orig_mod是编译模型的原始属性）
    #    原因：编译后的模型评测时可能有兼容问题，需用原始模型
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    
    # 5. 将模型切换为评测模式：关闭Dropout、BatchNorm等训练特有的层，保证评测结果稳定
    raw_model.eval()  
    
    # 6-8. 打印评测开始的分隔符和提示，方便日志阅读
    print("\n" + "="*60)
    print("开始 Benchmark 评测")
    print("="*60)
    
    # 9. 开始评测C3数据集（常识对话推理任务）
    try:
        # 10. 打印当前评测的数据集路径，方便排查数据路径错误
        print(f"评测 C3 数据集: {c3_path}")
        # 11. 调用eval_c3函数（外部定义），计算模型在C3数据集上的准确率
        #     输入：原始模型、tokenizer、C3数据集路径；输出：准确率（0-1之间）
        c3_acc = eval_c3(raw_model, tokenizer, c3_path)
        # 12. 将C3准确率存入结果字典，键名c3_accuracy，方便后续可视化/日志记录
        results['c3_accuracy'] = c3_acc
        # 13. 打印C3评测结果：保留4位小数（0.8567）和百分比（85.67%），直观展示
        print(f"✓ C3 Accuracy: {c3_acc:.4f} ({c3_acc*100:.2f}%)")
    # 14. 捕获C3评测过程中的所有异常（如数据路径错误、模型推理报错等）
    except Exception as e:
        # 15. 打印错误信息，方便定位问题（如“文件不存在”“维度不匹配”）
        print(f"✗ C3 evaluation failed: {e}")
        # 16. 异常时准确率记为0，避免后续代码因键不存在报错
        results['c3_accuracy'] = 0.0
    
    # 17. 开始评测XCOPA数据集（跨语言因果推理任务）
    try:
        # 18. 打印当前评测的数据集路径
        print(f"评测 XCOPA 数据集: {xcopa_path}")
        # 19. 调用eval_xcopa函数（外部定义），计算模型在XCOPA数据集上的准确率
        xcopa_acc = eval_xcopa(raw_model, tokenizer, xcopa_path)
        # 20. 将XCOPA准确率存入结果字典
        results['xcopa_accuracy'] = xcopa_acc
        # 21. 打印XCOPA评测结果，格式和C3一致
        print(f"✓ XCOPA Accuracy: {xcopa_acc:.4f} ({xcopa_acc*100:.2f}%)")
    # 22. 捕获XCOPA评测过程中的所有异常
    except Exception as e:
        # 23. 打印错误信息
        print(f"✗ XCOPA evaluation failed: {e}")
        # 24. 异常时准确率记为0
        results['xcopa_accuracy'] = 0.0
    
    # 25-26. 打印评测结束的分隔符，日志结构更清晰
    print("="*60 + "\n")
    
    # 27. 将模型恢复为训练模式：重新开启Dropout等层，不影响后续训练
    raw_model.train()  
    
    # 28. 返回包含两个数据集准确率的字典（如{'c3_accuracy':0.85, 'xcopa_accuracy':0.78}）
    return results