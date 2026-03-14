"""
SpongeBob 模型交互式对话脚本（简化版）
"""
# 1. 导入必备模块
import argparse  # 命令行参数解析模块，用于接收运行脚本时的参数（如模型路径、设备等）
import torch     # PyTorch 核心库，用于模型加载、张量运算
from transformers import AutoTokenizer, TextStreamer  # HuggingFace 工具：自动加载分词器、流式输出生成文本
from model.config import SpongeBobConfig  # 导入自定义的 SpongeBob 模型配置类
from model.model_spongebob_pro import SpongeBobForCausalLM  # 导入自定义的 SpongeBob 因果语言模型类

# 2. 主函数：脚本核心逻辑入口
def main():
    # 2.1 创建命令行参数解析器，定义脚本可接收的参数
    parser = argparse.ArgumentParser(description="SpongeBob模型交互对话")
    # 模型权重路径（必填，指定.pth格式的模型文件）
    parser.add_argument('--model_path', default='/root/autodl-tmp/spongebobpro/pretrain_768.pth', type=str, help="模型权重路径（.pth文件）")
    # 分词器路径（默认./tokenizer_15k，对应之前解决的tokenizer路径问题）
    parser.add_argument('--tokenizer_path', default='./tokenizer_15k/tokenizer_15k', type=str, help="Tokenizer路径（需指向含 tokenizer.json 的目录）")
    # 模型类型：pretrain（文本续写）/sft（对话），默认sft
    parser.add_argument('--model_type', default='sft', type=str, choices=['pretrain', 'sft'], help="模型类型：pretrain（文本续写）或 sft（对话）")
    # 模型隐藏层维度（默认768，需和训练时的配置一致）
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    # 模型隐藏层数量（默认12，需和训练时的配置一致）
    parser.add_argument('--num_hidden_layers', default=12, type=int, help="隐藏层数量")
    # 生成文本的最大长度（默认2048，控制回复的最长字数）
    parser.add_argument('--max_new_tokens', default=2048, type=int, help="最大生成长度")
    # 生成温度（0-1，越小越保守，回复越固定；越大越随机）
    parser.add_argument('--temperature', default=0.2, type=float, help="生成温度（0-1）")
    # 核采样阈值（0.7表示只从概率前70%的token中采样，减少无意义生成）
    parser.add_argument('--top_p', default=0.7, type=float, help="nucleus采样阈值")
    # 运行设备：自动检测CUDA（GPU），无则用CPU
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    # 多轮对话开关：不传则单轮（每轮独立），传则保留对话历史
    parser.add_argument('--multi_turn', action='store_true', help="保留对话历史（多轮）；不传则单轮，每轮独立")
    # 解析命令行传入的参数，赋值给args变量（后续所有参数都从args中取）
    args = parser.parse_args()
    
    # 2.2 自动推断模型类型（从模型路径文件名中识别，覆盖手动指定的类型）
    if 'pretrain' in args.model_path:  # 路径含pretrain → 文本续写模式
        args.model_type = 'pretrain'
    elif 'sft' in args.model_path:     # 路径含sft → 对话模式
        args.model_type = 'sft'
    
    # 2.3 加载分词器和模型（核心步骤）
    print(f'加载模型: {args.model_path}')  # 打印加载中的模型路径，方便排查
    # 加载分词器：从指定路径加载，对应之前解决的tokenizer依赖/权限问题
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # 初始化SpongeBob模型：先加载配置（隐藏层维度、数量），再初始化模型结构
    model = SpongeBobForCausalLM(SpongeBobConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers
    ))
    # 加载模型权重：从.pth文件读取权重，映射到指定设备（GPU/CPU）
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    # 模型设置为评估模式（关闭dropout等训练层），并移到指定设备
    model.eval().to(args.device)
    
    # 2.4 打印加载完成的提示信息，方便用户确认配置
    print(f'✅ 模型加载完成！设备: {args.device}')
    print(f'📝 模型类型: {args.model_type} ({"对话模式" if args.model_type == "sft" else "文本续写"})')
    print(f'📎 对话模式: {"多轮（保留历史）" if args.multi_turn else "单轮（每轮独立）"}\n')
    print('='*60)
    print('💬 开始对话 (输入 exit 退出)')
    print('='*60)
    
    # 2.5 初始化对话历史列表（仅多轮模式时使用）
    conversation = []  # 仅 multi_turn 时使用，存储[{"role": "user/assistant", "content": "文本"}]
    # 初始化流式输出器：让模型生成文本时逐字输出（而非一次性输出，提升交互体验）
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
    
    # 2.6 核心对话循环：持续接收用户输入，生成回复
    while True:
        # 接收用户输入，去除首尾空格
        user_input = input('\n👤 你: ').strip()
        
        # 退出条件：输入exit/quit/退出时终止循环
        if user_input.lower() in ['exit', 'quit', '退出']:
            print('👋 再见！')
            break
        
        # 空输入处理：跳过空行，重新等待输入
        if not user_input:
            continue
        
        # 2.7 按模型类型格式化输入文本
        if args.model_type == 'pretrain':
            # 文本续写模式：直接使用用户输入作为生成前缀
            formatted_input = user_input
            conversation = []  # 续写模式清空对话历史
        else:
            # SFT对话模式：按多轮/单轮规则构建对话格式
            if args.multi_turn:
                # 多轮模式：将当前输入追加到对话历史
                conversation.append({"role": "user", "content": user_input})
            else:
                # 单轮模式：重置对话历史，仅保留当前输入
                conversation = [{"role": "user", "content": user_input}]
            # 应用对话模板：将[{"role":..., "content":...}]转为模型能识别的格式（如<|user|>你好<|assistant|>）
            formatted_input = tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,  # 先不分词，仅格式化文本
                add_generation_prompt=True  # 添加生成提示符（如<|assistant|>），告诉模型该生成回复了
            )
        
        # 2.8 将格式化后的文本分词，转为模型可处理的张量（输入ID、注意力掩码），并移到指定设备
        inputs = tokenizer(formatted_input, return_tensors="pt").to(args.device)
        
        # 2.9 生成模型回复（核心生成逻辑）
        print('🧽 SpongeBob: ', end='', flush=True)  # 打印回复前缀，不换行
        with torch.no_grad():  # 禁用梯度计算，节省显存、提升速度（评估模式必备）
            generated_ids = model.generate(
                inputs=inputs["input_ids"],  # 输入文本的token ID
                attention_mask=inputs["attention_mask"],  # 注意力掩码（避免模型关注padding token）
                max_new_tokens=args.max_new_tokens,  # 最大生成新token数
                do_sample=True,  # 开启采样（配合temperature/top_p，生成更自然的文本）
                streamer=streamer,  # 流式输出（逐字打印回复）
                pad_token_id=tokenizer.eos_token_id,  # padding token ID（用结束符填充）
                eos_token_id=tokenizer.eos_token_id,  # 结束符token ID（生成到该ID停止）
                top_p=args.top_p,  # 核采样阈值
                temperature=args.temperature,  # 生成温度
                repetition_penalty=1.2  # 重复惩罚（1.2表示轻微惩罚重复文本，避免回复啰嗦）
            )
        
        # 2.10 解码生成的token ID，转为可读文本（跳过输入部分，只取生成的回复）
        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):],  # 截取生成的部分（排除输入的token）
            skip_special_tokens=False  # 不跳过特殊token（如<|assistant|>）
        )
        # 2.11 多轮对话模式：将模型回复追加到对话历史，供下一轮使用
        if args.model_type == 'sft' and args.multi_turn:
            conversation.append({"role": "assistant", "content": response})

# 3. 脚本入口：只有直接运行该脚本时，才执行main函数
if __name__ == "__main__":
    main()