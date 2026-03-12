#!/usr/bin/env python3
'''
dataset 目录负责从原始语料到可供模型训练使用的数据集的“全流程”：先预处理生成二进制数据，再提供加载这些数据的 PyTorch 数据集类。
'''
"""
把原始的 JSONL 文本语料，离线预处理成适合大模型预训练的 .bin + .meta 二进制数据集文件。
"""
import os
import sys
import json
import argparse
import tempfile
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count

# 1. 定义全局变量：供子进程调用，避免每个子进程重复加载tokenizer（节省内存+加速）
#    _tokenizer：存储分词器实例
#    _eos_id：存储分词器的结束符token id（如<eos>对应的数字）
_tokenizer = None
_eos_id = None

def _init_worker(tokenizer_path):
    """初始化子进程的tokenizer
    作用：在每个子进程启动时只加载一次tokenizer，而非处理每行数据都加载
    """
    # 2. 声明使用全局变量（否则子进程内的赋值仅作用于局部）
    global _tokenizer, _eos_id
    # 3. 从指定路径加载预训练的分词器（如GPT2/LLaMA的tokenizer）
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # 4. 获取结束符（eos）的token id，后续给每条文本追加结束符
    _eos_id = _tokenizer.eos_token_id

def _tokenize_line(line):
    """处理单行数据：解析JSONL行→提取文本→分词→追加结束符
    Args:
        line: JSONL格式的单行字符串（如{"text": "今天天气好"}）
    Returns:
        分词后的token id列表（失败返回空列表）
    """
    try:
        # 5. 去除行首尾的空白字符（换行/空格/制表符），避免解析JSON出错
        line = line.strip()
        # 6. 如果处理后是空行，直接返回空列表（跳过无效行）
        if not line:
            return []
        # 7. 解析单行JSON字符串为字典（适配JSONL格式的数据集）
        data = json.loads(line)
        # 8. 从字典中提取"text"字段的文本内容（无则返回空字符串）
        text = data.get('text', '')
        # 9. 如果提取的文本为空，返回空列表（跳过无文本的样本）
        if not text:
            return []
        # 10. 对文本进行分词：转为token id列表
        #     add_special_tokens=False：不自动添加bos/eos等特殊符（后续手动加eos）
        tokens = _tokenizer.encode(text, add_special_tokens=False)
        # 11. 给每条文本的token列表追加结束符id（符合大模型训练格式要求）
        tokens.append(_eos_id)
        # 12. 返回处理后的token id列表（用于模型训练）
        return tokens
    except Exception as e:
        # 13. 捕获所有异常（如JSON解析失败、文本编码错误等），静默返回空列表
        #     目的：避免单条数据处理失败导致整个多进程分词流程中断
        return []

def preprocess(input_path, output_path, tokenizer_path, seq_len=512, num_workers=None):
    """
    预处理：tokenize + 拼接 + 切分 + 保存为.bin
    
    输出文件：
    - output_path.bin: 所有token数据 (int16格式)
    - output_path.meta: 元信息 (json格式)
    """
    # 1. 设置默认进程数：如果未指定，使用CPU核心数（最多32个，避免进程过多导致卡顿）
    if num_workers is None:
        num_workers = cpu_count()  # 最多32个进程
    
    # 2-9. 打印预处理任务的关键信息，方便日志查看和问题排查
    print(f"{'='*60}")
    print(f"预训练数据预处理")
    print(f"{'='*60}")
    print(f"输入: {input_path}")          # 输入数据集路径（JSONL格式）
    print(f"输出: {output_path}.bin")     # 输出token二进制文件路径
    print(f"Tokenizer: {tokenizer_path}") # 分词器路径
    print(f"序列长度: {seq_len}")         # 模型训练的固定序列长度（如512）
    print(f"进程数: {num_workers}")       # 多进程数
    print(f"{'='*60}\n")
    
    # 10-16. 加载分词器并获取核心信息（为后续tokenize做准备）
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)  # 加载预训练分词器
    vocab_size = len(tokenizer)                                # 获取词表大小（后续写入元信息）
    eos_id = tokenizer.eos_token_id                            # 获取结束符token id（文本拼接用）
    print(f"词表大小: {vocab_size}")                           # 打印词表大小，确认分词器加载正常
    print(f"EOS token: {eos_id} ('{tokenizer.decode([eos_id])}')\n")  # 打印EOS的id和对应的字符，验证正确性
    
    # 17-21. 统计输入文件的总行数（样本数），用于进度展示和后续统计
    print("步骤1: 统计样本数...")
    with open(input_path, 'r', encoding='utf-8') as f:
        # 逐行遍历文件，统计总行数（即样本数），sum(1 for _ in f) 是高效统计行数的方式
        num_samples = sum(1 for _ in f)
    print(f"样本数: {num_samples:,}\n")  # 格式化输出（如100000→100,000），更易读
    
    # 22-23. 准备多进程tokenize（核心预处理步骤的开头）
    print(f"步骤2: Tokenizing (使用 {num_workers} 个进程)...")
    
    #作用：不一次性把整个文件读进内存，而是按行生成，适合超大数据集。
    def line_generator():
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line
    
    # 作用：创建一个二进制临时文件，用来边处理边写入 token，避免内存爆炸。
    temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.tmp')
    temp_path = temp_file.name
    
    total_tokens = 0
    buffer = []
    BUFFER_SIZE = 10_000_000  # 每1000万个token写一次磁盘（~20MB）
    
    try:
        with Pool(num_workers, initializer=_init_worker, initargs=(tokenizer_path,)) as pool:
            for tokens in tqdm(pool.imap(_tokenize_line, line_generator(), chunksize=100), total=num_samples):
                if tokens:
                    buffer.extend(tokens)
                    total_tokens += len(tokens)
                    
                    # 缓冲区满了，写入磁盘
                    if len(buffer) >= BUFFER_SIZE:
                        np.array(buffer, dtype=np.uint16).tofile(temp_file)
                        buffer = []
        
        # 写入剩余数据
        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(temp_file)
            buffer = []
        
        temp_file.close()
        
        print(f"\n总tokens: {total_tokens:,}")
        print(f"平均长度: {total_tokens/num_samples:.1f} tokens/sample\n")
        
        # 切分成固定长度chunks
        print(f"步骤3: 切分成 {seq_len} 长度的chunks...")
        num_chunks = total_tokens // seq_len
        dropped = total_tokens % seq_len
        
        # 从临时文件读取并切分（内存映射）
        all_tokens = np.fromfile(temp_path, dtype=np.uint16)
        all_tokens = all_tokens[:num_chunks * seq_len]  # 只保留完整chunks
        arr = all_tokens.reshape(-1, seq_len)
        
        
        print(f"Chunks数: {num_chunks:,}")
        print(f"丢弃tokens: {dropped} ({dropped/total_tokens*100:.3f}%)")
        print(f"数组形状: {arr.shape}")
        print(f"内存占用: {arr.nbytes / (1024**3):.2f} GB\n")
        
        # 保存为.bin文件
        print(f"步骤4: 保存为二进制文件...")
        bin_path = f"{output_path}.bin"
        arr.tofile(bin_path)
        print(f"✓ 已保存: {bin_path} ({os.path.getsize(bin_path)/(1024**3):.2f} GB)")
        
        # 保存元信息
        meta = {
            "vocab_size": vocab_size,
            "seq_len": seq_len,
            "num_chunks": num_chunks,
            "total_tokens": total_tokens,
            "num_samples": num_samples,
            "dropped_tokens": dropped,
            "dtype": "uint16",
            "shape": list(arr.shape),
        }
        
        meta_path = f"{output_path}.meta"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"✓ 已保存: {meta_path}\n")
        
        print(f"{'='*60}")
        print(f"预处理完成！")
        print(f"{'='*60}")
        print(f"训练时使用: --data_path {bin_path}")
        print(f"{'='*60}\n")
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预训练数据预处理")
    parser.add_argument("--input", type=str, required=True, help="输入jsonl文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出文件前缀（不含扩展名）")
    parser.add_argument("--tokenizer", type=str, default="../tokenizer_15k", help="tokenizer路径")
    parser.add_argument("--seq_len", type=int, default=512, help="序列长度")
    parser.add_argument("--num_workers", type=int, default=None, help="进程数（默认=CPU核心数，最多32）")
    args = parser.parse_args()
    
    preprocess(args.input, args.output, args.tokenizer, args.seq_len, args.num_workers)   