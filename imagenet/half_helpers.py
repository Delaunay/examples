import torch


global_use_gpu = False
global_use_half = False


def set_use_gpu(val):
    global global_use_gpu
    global_use_gpu = torch.cuda.is_available() and val


def set_use_half(val):
    global global_use_half
    global_use_half  = val


def use_gpu() -> bool:
    global global_use_gpu
    return global_use_gpu


def use_half() -> bool:
    global global_use_gpu
    global global_use_half

    return global_use_gpu and global_use_half


def enable_cuda(object):
    if use_gpu():
        return object.cuda(non_blocking=True)

    return object


def enable_half(object):
    if not use_gpu():
        return object

    if not use_half():
        return object.cuda()

    # F32 Tensor
    try:
        if object.dtype == torch.float32:
            return object.cuda(non_blocking=True).half(non_blocking=True)
    except:
        # Not a tensor
        return object.cuda().half()

    # different type
    return object.cuda(non_blocking=True)


class OptimizerAdapter:
    def __init__(self, optimizer, half=False, *args, **kwargs):
        if half:
            import apex.fp16_utils.fp16_optimizer as apex_optimizer

            self.optimizer = apex_optimizer.FP16_Optimizer(optimizer, *args, **kwargs)
        else:
            self.optimizer = optimizer

        self.half = half

    def backward(self, loss):
        if loss is None:
            raise RuntimeError('')

        if self.half:
            self.optimizer.backward(loss)
        else:
            loss.backward()

        return self.optimizer

    def step(self):
        return self.optimizer.step()

    def zero_grad(self):
        return self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups
