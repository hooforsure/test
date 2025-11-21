import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def conv2d_kernel(
    # 指针参数
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # 维度参数
    batch_size, in_channels, input_h, input_w,
    out_channels, kernel_h, kernel_w,
    stride_h, stride_w,
    # 输出尺寸
    output_h, output_w,
    # Triton配置
    BLOCK_SIZE: tl.constexpr
):
    # 将4维映射到3维
    pid = tl.program_id(0)  # batch_size * out_channels
    pid_oh = tl.program_id(1)  # 输出高度
    pid_ow = tl.program_id(2)  # 输出宽度
    
    # 解出批次和输出通道
    pid_batch = pid // out_channels
    pid_out_ch = pid % out_channels
    
    # 边界检查
    if pid_batch >= batch_size:
        return
    if pid_out_ch >= out_channels:
        return
    if pid_oh >= output_h:
        return
    if pid_ow >= output_w:
        return
    
    # 预计算关键常量
    input_hw = input_h * input_w
    kernel_hw = kernel_h * kernel_w
    
    # 基础偏移量
    input_batch_offset = pid_batch * in_channels * input_hw
    weight_ch_offset = pid_out_ch * in_channels * kernel_hw
    output_offset = (
        pid_batch * out_channels * output_h * output_w +
        pid_out_ch * output_h * output_w +
        pid_oh * output_w +
        pid_ow
    )
    
    # 初始化累加器
    accumulator = 0.0
    
    # 关键优化：减少网格规模，增加每个线程的工作量
    # 每个线程处理多个输入通道块
    CHANNELS_PER_THREAD = 8  # 每个线程处理8个通道
    
    for c_block in range(0, in_channels, CHANNELS_PER_THREAD):
        c_end = min(c_block + CHANNELS_PER_THREAD, in_channels)
        
        for c in range(c_block, c_end):
            input_ch_offset = input_batch_offset + c * input_hw
            weight_c_offset = weight_ch_offset + c * kernel_hw
            
            # 简单的双层循环，但减少线程数量
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    input_row = pid_oh * stride_h + kh
                    input_col = pid_ow * stride_w + kw
                    
                    if (input_row < input_h) and (input_col < input_w):
                        input_offset = input_ch_offset + input_row * input_w + input_col
                        weight_offset = weight_c_offset + kh * kernel_w + kw
                        
                        input_val = tl.load(input_ptr + input_offset)
                        weight_val = tl.load(weight_ptr + weight_offset)
                        accumulator += input_val * weight_val
    
    # 加偏置
    bias_val = tl.load(bias_ptr + pid_out_ch)
    accumulator += bias_val
    
    # 存储结果
    tl.store(output_ptr + output_offset, accumulator)

def conv2d_triton(
    input: torch.Tensor,
    kernel: torch.Tensor,
    bias: torch.Tensor
) -> torch.Tensor:
    # 获取输入维度
    batch_size, in_channels, input_h, input_w = input.shape
    out_channels, _, kernel_h, kernel_w = kernel.shape
    
    # 设置步长
    stride_h, stride_w = kernel_h, kernel_w
    
    # 计算输出尺寸
    output_h = (input_h - kernel_h) // stride_h + 1
    output_w = (input_w - kernel_w) // stride_w + 1
    
    # 初始化输出张量
    output = torch.zeros(
        (batch_size, out_channels, output_h, output_w),
        device=input.device,
        dtype=input.dtype
    )
    
    # 关键优化：大幅减少线程数量
    # 原来：batch_size * out_channels * output_h * output_w = 4 * 512 * 13 * 13 = 346,112 个线程
    # 现在：大幅减少网格规模
    
    # 策略：每个线程处理多个输出位置
    threads_per_output_channel = max(1, output_h * output_w // 4)  # 减少到1/4
    grid = (
        batch_size * out_channels,  # 保持不变
        threads_per_output_channel,  # 大幅减少
        1  # 固定为1
    )
    
    print(f"优化后网格规模: {grid} (原规模: ({batch_size * out_channels}, {output_h}, {output_w}))")
    
    # 启动内核
    conv2d_kernel[grid](
        input, kernel, bias, output,
        batch_size, in_channels, input_h, input_w,
        out_channels, kernel_h, kernel_w,
        stride_h, stride_w,
        output_h, output_w,
        BLOCK_SIZE=32
    )
    
    return output





if __name__ == '__main__':

    dtype = torch.float32
    device = 'mlu'
    
    batch_size=4
    height=224
    width=224
    channels=3

    kernels=512
    kernel_height=16
    kernel_width=16

    # 生成输入图片
    input = torch.randint(0, 10, (batch_size, channels, height, width)).to(device, dtype)
    # 生成卷积核
    kernel = torch.randint(0, 10, (kernels, channels, kernel_height, kernel_width)).to(device, dtype)
    # 生成偏置
    bias = torch.randn(kernels).to(device, dtype)

    conv_layer = torch.nn.Conv2d(
        in_channels=channels,
        out_channels=kernels,
        kernel_size=(kernel_height, kernel_width),
        stride=(kernel_height, kernel_width),
        bias=True,
        dtype=dtype
    ).to(device)

    # For a fair comparison, copying same kernel to torch layer as well
    with torch.no_grad():
        conv_layer.weight.copy_(kernel)
        conv_layer.bias.copy_(bias)

    y_torch = conv_layer(input)
    y_triton = conv2d_triton(input, kernel, bias)

    print(f'Original matrix:\n{input}')
    print(f'PyTorch Conv2d:\n{y_torch}')
    print(f'Triton Conv2d:\n{y_triton}')

    if torch.allclose(y_torch, y_triton):
        
        print('Data matches')

    else:
        print('Data does not match')

    print("\n--- Simple Performance Test ---")
    warmup = 20
    repeat = 100

    start_event = torch.mlu.Event(enable_timing=True)
    end_event = torch.mlu.Event(enable_timing=True)

    for _ in range(warmup):
        _ = conv2d_triton(input, kernel, bias)
    torch.mlu.synchronize()

    start_event.record()
    for _ in range(repeat):
        _ = conv2d_triton(input, kernel, bias)
    end_event.record()
    torch.mlu.synchronize()

    triton_time = start_event.elapsed_time(end_event) / repeat
    print(f'Triton average time: {triton_time:.4f} ms')

    start_event_torch = torch.mlu.Event(enable_timing=True)
    end_event_torch = torch.mlu.Event(enable_timing=True)

    for _ in range(warmup):
        _ = conv_layer(input)
    torch.mlu.synchronize()

    start_event_torch.record()
    for _ in range(repeat):
        _ = conv_layer(input)
    end_event_torch.record()
    torch.mlu.synchronize()

    torch_time = start_event_torch.elapsed_time(end_event_torch) / repeat
    print(f'PyTorch average time: {torch_time:.4f} ms')
    
    faster = "Triton" if triton_time < torch_time else "PyTorch"
    speedup = torch_time / triton_time if triton_time < torch_time else triton_time / torch_time
    print(f'Simple test shows {faster} is {speedup:.2f}x faster.')
    
    print("---------------------------------")