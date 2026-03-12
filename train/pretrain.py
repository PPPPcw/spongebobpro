"""
单卡预训练脚本（无 DDP）
用法: python pretrain_without_ddp.py [args]
"""
import os
import sys

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

__package__ = "train"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader
from model.config import SpongeBobConfig
from model.model_spongebob_pro import SpongeBobForCausalLM
from dataset.pretrain_dataset import PretrainDataset
from utils import get_lr, Logger, SkipBatchSampler
from benchmark.evaluator import run_benchmark

warnings.filterwarnings('ignore')

'''
告诉函数「当前训练到哪一轮（epoch）、本轮有多少批数据（iters）、从哪一批开始续训（start_step）」；
告诉函数「数据从哪取（loader）、学习率怎么调（total_steps/warmup_steps）」；
告诉函数「模型存在哪（full_save_dir）、训练指标要不要可视化（swanlab）」；
'''
def train_epoch(epoch, loader, iters, start_step=0, swanlab=None, total_steps=None, warmup_steps=None, full_save_dir=None):
    #1. 初始化 + 遍历数据：逐批次取数据
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
    #2.动态调整学习率：按训练步数改 LR
        current_step = epoch * iters + step
        lr = get_lr(current_step, total_steps, args.learning_rate, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #3.前向计算 + 损失计算：混合精度加速
        with autocast_ctx:  # 混合精度训练（bfloat16/fp16），提速+省显存
            res = model(input_ids, labels=labels)# 调用模型前向传播，算logits和loss
            loss = res.loss / args.accumulation_steps # 梯度累积：把损失均分（后续反向传播用）

        scaler.scale(loss).backward()  # 放大loss，再反向传播算梯度（避免下溢）
    #4.反向传播 + 参数更新：核心训练逻辑
        # 梯度累积步数到了，才更新参数
        #梯度更新时：先把梯度缩回去，再更新参数
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 把放大的梯度缩回原大小
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪：防止梯度爆炸
            scaler.step(optimizer)  # 更新模型参数
            scaler.update() # 调整放大倍数（自适应）
            optimizer.zero_grad(set_to_none=True)  # 清空梯度，准备下一批数据

        global_step = epoch * iters + step
    #5. 日志记录：打印训练状态
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if swanlab:
                swanlab.log({"loss": current_loss, "learning_rate": current_lr, "eta_time": eta_min}, step=global_step)

        # 6. 保存模型：存 checkpoint（续训 / 复用）
        if global_step % args.save_interval == 0 or step == iters - 1:
            model.eval()
            ckp_dir = f'{full_save_dir}/global_step_{global_step}'
            os.makedirs(ckp_dir, exist_ok=True)
            raw_model = getattr(model, '_orig_mod', model)
            state_dict = {k: v.half().cpu() for k, v in raw_model.state_dict().items()}
            torch.save(state_dict, f'{ckp_dir}/{args.save_weight}_{lm_config.hidden_size}.pth')
            torch.save({
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'step': step,
                'global_step': global_step,
                'swanlab_id': getattr(swanlab, 'id', None) if swanlab else None
            }, f'{ckp_dir}/resume.pth')
            Logger(f'Saved checkpoint: {ckp_dir}')
            model.train()

        # 7.Benchmark 评测 评测效果：验证模型性能
        if args.eval_bench == 1 and tokenizer is not None and global_step % args.eval_interval == 0:
            model.eval()
            c3_path = '测试集地址'
            xcopa_path = '测试集地址'
            eval_results = run_benchmark(model, tokenizer, c3_path, xcopa_path)
            if swanlab_run:
                swanlab_run.log(eval_results, step=global_step)
            Logger(f'Benchmark results: {eval_results}')
            model.train()
        #8 清理内存：避免显存泄漏
        del input_ids, labels, res, loss   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpongeBob Pretraining (Single GPU)")
    parser.add_argument("--save_dir", type=str, default="../pretrain_out", help="模型保存根目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=12, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="序列长度")
    parser.add_argument("--data_path", type=str, default="{你的文件路径}", help="预处理后的.bin文件路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_swanlab", type=int, default=1, choices=[0, 1], help="是否使用swanlab（0=否，1=是）")
    parser.add_argument("--swanlab_project", type=str, default="SpongeBob-Pretrain", help="swanlab项目名")
    parser.add_argument("--use_compile", default=1, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--eval_bench", default=1, type=int, choices=[0, 1], help="是否评测benchmark（0=否，1=是）")
    parser.add_argument("--eval_interval", type=int, default=100, help="评测间隔步数")
    args = parser.parse_args()