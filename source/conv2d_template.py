import torch
import triton
import triton.language as tl
from typing import Tuple



# TODO: Implement the conv2d kernel
@triton.jit
def conv2d_kernel(
    # Define kernel parameters here, e.g., input pointer, weight pointer, output pointer
):
    # TODO: Use tl.load and tl.store to implement the convolution computation
    pass


# TODO: Implement the wrapper function
def conv2d_triton(
    input: torch.Tensor,
    kernel: torch.Tensor,
    bias: torch.Tensor
) -> torch.Tensor:
    """
    Execute conv2d on input using the Triton kernel.

    """
    # TODO: Initialize the output tensor
    output = torch.zeros( # shape, device, dtype
    )

    # TODO: Define grid dimensions and launch the kernel
    # grid = (N, K, OH, OW)
    # conv2d_kernel[grid](...)

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

    input = torch.randint(0, 10, (batch_size, channels, height, width)).to(device, dtype)
    kernel = torch.randint(0, 10, (kernels, channels, kernel_height, kernel_width)).to(device, dtype)
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