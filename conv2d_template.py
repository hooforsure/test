import torch
import triton
import triton.language as tl
from typing import Tuple



# TODO: Implement the conv2d kernel
@triton.jit
def conv2d_kernel(
        # 指针参数
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        # 维度参数
        batch_size, in_channels, input_h, input_w,
        out_channels, kernel_h, kernel_w,
        stride_h, stride_w,
        # Triton配置
        BLOCK_SIZE: tl.constexpr
):
    # 获取当前工人的ID - 对应输出张量的4个维度
    pid_batch = tl.program_id(0)  # 批次维度 [0, batch_size-1]
    pid_out_ch = tl.program_id(1)  # 输出通道 [0, out_channels-1]
    pid_oh = tl.program_id(2)  # 输出高度 [0, output_h-1]
    pid_ow = tl.program_id(3)  # 输出宽度 [0, output_w-1]

    # 计算输出尺寸
    output_h = (input_h - kernel_h) // stride_h + 1
    output_w = (input_w - kernel_w) // stride_w + 1

    # 边界检查：如果这个工人负责的位置超出范围，直接返回
    if pid_batch >= batch_size or pid_out_ch >= out_channels or pid_oh >= output_h or pid_ow >= output_w:
        return

    # 初始化累加器
    accumulator = 0.0

    # 计算卷积：遍历输入通道和卷积核空间维度
    for c in range(in_channels):  # 输入通道循环
        for kh in range(kernel_h):  # 卷积核高度循环
            for kw in range(kernel_w):  # 卷积核宽度循环
                # 计算输入图像中的对应位置
                input_row = pid_oh * stride_h + kh
                input_col = pid_ow * stride_w + kw

                # 边界检查
                if input_row < input_h and input_col < input_w:
                    # 计算输入张量的内存偏移量
                    input_offset = (
                            pid_batch * in_channels * input_h * input_w +  # 批次偏移
                            c * input_h * input_w +  # 通道偏移
                            input_row * input_w +  # 行偏移
                            input_col  # 列偏移
                    )

                    # 计算权重张量的内存偏移量
                    weight_offset = (
                            pid_out_ch * in_channels * kernel_h * kernel_w +  # 输出通道偏移
                            c * kernel_h * kernel_w +  # 输入通道偏移
                            kh * kernel_w +  # 卷积核行偏移
                            kw  # 卷积核列偏移
                    )

                    # 从全局内存加载数据
                    input_val = tl.load(input_ptr + input_offset)
                    weight_val = tl.load(weight_ptr + weight_offset)

                    # 乘积累加
                    accumulator += input_val * weight_val

    # 加载偏置并加到累加器
    bias_val = tl.load(bias_ptr + pid_out_ch)
    accumulator += bias_val

    # 计算输出张量的内存偏移量
    output_offset = (
            pid_batch * out_channels * output_h * output_w +  # 批次偏移
            pid_out_ch * output_h * output_w +  # 输出通道偏移
            pid_oh * output_w +  # 输出行偏移
            pid_ow  # 输出列偏移
    )

    # 将结果存储到输出张量
    tl.store(output_ptr + output_offset, accumulator)


# TODO: Implement the wrapper function
def conv2d_triton(
        input: torch.Tensor,
        kernel: torch.Tensor,
        bias: torch.Tensor
) -> torch.Tensor:
    """
    使用Triton实现2D卷积
    """
    # 获取输入张量维度
    batch_size, in_channels, input_h, input_w = input.shape
    out_channels, _, kernel_h, kernel_w = kernel.shape

    # 设置步长（根据作业要求使用卷积核尺寸作为步长）
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

    # 定义网格维度 - 对应输出张量的4个维度
    grid = (batch_size, out_channels, output_h, output_w)

    # 启动卷积内核
    conv2d_kernel[grid](
        input, kernel, bias, output,  # 数据指针
        batch_size, in_channels, input_h, input_w,  # 输入维度
        out_channels, kernel_h, kernel_w,  # 卷积核维度
        stride_h, stride_w,  # 步长
        BLOCK_SIZE=32  # 块大小（可调整）
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