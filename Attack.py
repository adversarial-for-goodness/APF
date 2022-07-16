import torch
import torch.nn.functional as F
from enum import Enum
import scipy.stats as st
import numpy as np

class NormType(Enum):
    Linf = 0
    L2 = 1

def clamp_by_l2(x, max_norm):
    norm = torch.norm(x, dim=(1,2,3), p=2, keepdim=True)
    factor = torch.min(max_norm / norm, torch.ones_like(norm))
    return x * factor

def random_init(x, norm_type, epsilon):
    delta = torch.zeros_like(x)
    if norm_type == NormType.Linf:
        delta.data.uniform_(-1.0, 1.0)
        delta.data = delta.data * epsilon
    elif norm_type == NormType.L2:
        delta.data.uniform_(-1.0, 1.0)
        delta.data = delta.data - x
        delta.data = clamp_by_l2(delta.data, epsilon)
    return delta

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel

class Attack():
    def __init__(self, net, loss_fn, norm_type=NormType.Linf, random_init=True, *args, **kwargs):
        self.net = net
        self.loss_fn = loss_fn
        self.norm_type = norm_type
        self.random_init = random_init
        self.preprocess = kwargs.get('preprocess')

        self.bounding = kwargs.get('bounding')
        if self.bounding is None:
            self.bounding = (-1, 1)

    def input_diversity(self, x):
        if not hasattr(self, 'diversity_prob'):
            return x

    def run(self, x, labels, epsilon, num_iters, targeted=False):
        if self.random_init:
            delta = random_init(x, self.norm_type, epsilon)
        else:
            delta = torch.zeros_like(x)

        if hasattr(self, 'kernel'):
            self.kernel = self.kernel.to(x.device)

        if hasattr(self, 'grad'):
            self.grad = torch.zeros_like(x)

        if targeted:
            scaler = -1
        else:
            scaler = 1

        epsilon_per_iter = epsilon / num_iters * 1.25

        for i in range(num_iters):
            delta = delta.detach()
            delta.requires_grad = True
            x_diversity = self.input_diversity(x + delta)

            if self.preprocess is not None:
                x_diversity = self.preprocess(x_diversity)

            _, loss = self.net_forward(x_diversity, labels)

            grad = self.get_grad(delta, loss)

            grad = self.normalize(grad)

            delta = delta.data + epsilon_per_iter * grad * scaler

            # constraint 1: epsilon
            delta = self.project(delta, epsilon)
            # constraint 2: image range
            delta = torch.clamp(x + delta, *self.bounding) - x

        return x + delta, delta

    def net_forward(self, x, y):
        logits = self.net(x)
        loss = self.loss_fn(logits, y)
        return logits, loss

    def get_grad(self, delta, loss):
        loss.backward()
        grad = delta.grad.clone()
        return grad

    def project(self, delta, epsilon):
        if self.norm_type == NormType.Linf:
            return torch.clamp(delta, -epsilon, epsilon)
        elif self.norm_type == NormType.L2:
            return clamp_by_l2(delta, epsilon)

    def normalize(self, grad):
        if self.norm_type == NormType.Linf:
            return torch.sign(grad)
        elif self.norm_type == NormType.L2:
            return grad / torch.norm(grad, dim=(1, 2, 3), p=2, keepdim=True)

    def __call__(self, x, labels, epsilon, num_iters, targeted=False):
        return self.run(x, labels, epsilon, num_iters, targeted)

class DI_Attack(Attack):
    def __init__(self, net, loss_fn, norm_type=NormType.Linf, random_init=True, resize_rate=1.10, diversity_prob=0.3, *args, **kwargs):
        super(DI_Attack, self).__init__(net, loss_fn, norm_type, random_init, *args, **kwargs)
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob

    def input_diversity(self, x):
        assert self.resize_rate >= 1.0
        assert self.diversity_prob >= 0.0 and self.diversity_prob <= 1.0

        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)
        # print(img_size, img_resize, resize_rate)
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
        padded = F.interpolate(padded, size=[img_size, img_size])
        ret = padded if torch.rand(1) < self.diversity_prob else x
        return ret


class MI_Attack(Attack):
    def __init__(self, net, loss_fn, norm_type=NormType.Linf, random_init=True, momentum=0.9, *args, **kwargs):
        super(MI_Attack, self).__init__(net, loss_fn, norm_type, random_init, *args, **kwargs)
        self.momentum = momentum

    def get_grad(self, delta, loss):
        loss.backward()

        if not hasattr(self, 'grad'):
            self.grad = torch.zeros_like(delta)

        grad = delta.grad.clone()
        self.grad = self.grad * self.momentum + grad
        return self.grad


class TI_Attack(Attack):
    def __init__(self, net, loss_fn, norm_type=NormType.Linf, random_init=True, kernlen=15, nsig=3, *args, **kwargs):
        super(TI_Attack, self).__init__(net, loss_fn, norm_type, random_init, *args, **kwargs)
        kernel = gkern(kernlen, nsig).astype(np.float32)
        kernel = np.stack([kernel, kernel, kernel])
        kernel = np.expand_dims(kernel, 1)
        self.kernel = torch.from_numpy(kernel)
        self.kernlen = kernlen

    def get_grad(self, delta, loss):
        loss.backward()

        grad = delta.grad.clone()
        grad = F.conv2d(grad, self.kernel, padding=self.kernlen // 2, groups=3)
        grad = grad / torch.mean(grad.abs(), dim=(1,2,3), keepdim=True)

        return grad

class SI_Attack(Attack):
    def __init__(self, net, loss_fn, norm_type=NormType.Linf, random_init=True, *args, **kwargs):
        super(SI_Attack, self).__init__(net, loss_fn, norm_type, random_init, *args, **kwargs)

    def net_forward(self, x, y):
        loss = 0
        for i in range(5):
            logit = self.net(x * 0.5**i)
            loss += self.loss_fn(logit, y)
        return None, loss

class ADMIX_Attack(Attack):
    def __init__(self, net, loss_fn, norm_type=NormType.Linf, random_init=True, portion=0.2, repeat=3, *args, **kwargs):
        super(ADMIX_Attack, self).__init__(net, loss_fn, norm_type, random_init, *args, **kwargs)
        self.portion = portion
        self.repeat = repeat

    def admix(self, x):
        return torch.cat([(x + self.portion * x[torch.randperm(x.shape[0])]) for _ in range(self.repeat)], dim=0)

    def net_forward(self, x, y):
        x_mix = self.admix(x)
        logit = self.net(x_mix)
        if y.ndim == 2:
            loss = self.loss_fn(logit, y.repeat([self.repeat, 1]))
        else:
            loss = self.loss_fn(logit, y.repeat([self.repeat]))
        return logit, loss

class DI_MI_Attack(DI_Attack, MI_Attack):
    def __init__(self, net, loss_fn, norm_type=NormType.Linf, random_init=True, resize_rate=1.10, diversity_prob=0.3, momentum=0.9, *args, **kwargs):
        super(DI_MI_Attack, self).__init__(net, loss_fn, norm_type, random_init, resize_rate, diversity_prob, momentum, *args, **kwargs)

    def input_diversity(self, x):
        return DI_Attack.input_diversity(self, x)

    def get_grad(self, delta, loss):
        return MI_Attack.get_grad(self, delta, loss)


class TIDIM_Attack(DI_Attack, MI_Attack, TI_Attack):
    def __init__(self, net, loss_fn, norm_type=NormType.Linf, random_init=True, resize_rate=1.10, diversity_prob=0.3, momentum=0.9, kernlen=15, nsig=3, *args, **kwargs):
        super(TIDIM_Attack, self).__init__(net, loss_fn, norm_type, random_init, resize_rate, diversity_prob, momentum, kernlen, nsig, *args, **kwargs)

    def input_diversity(self, x):
        return DI_Attack.input_diversity(self, x)

    def get_grad(self, delta, loss):
        loss.backward()

        if not hasattr(self, 'grad'):
            self.grad = torch.zeros_like(delta)

        grad = delta.grad.clone()
        grad = F.conv2d(grad, self.kernel, padding=self.kernlen // 2, groups=3)
        grad = grad / torch.mean(grad.abs(), dim=(1,2,3), keepdim=True)

        self.grad = self.grad * self.momentum + grad
        return self.grad


class ADMIX_DI_MI_Attack(ADMIX_Attack, DI_Attack, MI_Attack):
    def __init__(self, net, loss_fn, norm_type=NormType.Linf, random_init=True,
                 portion=0.2, repeat=3, resize_rate=1.10, diversity_prob=0.3, momentum=0.9, *args, **kwargs):
        super(ADMIX_DI_MI_Attack, self).__init__(net, loss_fn, norm_type, random_init, portion, repeat, resize_rate, diversity_prob, momentum, *args, **kwargs)

    def net_forward(self, x, y):
        return ADMIX_Attack.net_forward(self, x, y)

    def get_grad(self, delta, loss):
        return MI_Attack.get_grad(self, delta, loss)


class DI_Ensemble_Attack(DI_Attack):
    def net_forward(self, x, y, beta=0):
        loss = 0
        net0, net1 = self.net[0], self.net[1]
        logits = net0(x)
        with torch.no_grad():
            target = net0(y).detach()
        loss += self.loss_fn(logits, target)
        logits = net1(x)
        with torch.no_grad():
            target = net1(y).detach()
        loss += self.loss_fn(logits, target) * beta
        return None, loss


class MI_Ensemble_Attack(MI_Attack):
    def net_forward(self, x, y):
        loss = 0
        net0, net1 = self.net[0], self.net[1]
        logits = net0(x)
        with torch.no_grad():
            target = net0(y).detach()
        loss += self.loss_fn(logits, target)
        logits = net1(x)
        with torch.no_grad():
            target = net1(y).detach()
        loss += self.loss_fn(logits, target) * self.beta
        return None, loss


class SI_Ensemble_Attack(SI_Attack):
    def net_forward(self, x, y):
        loss = 0
        net0, net1 = self.net[0], self.net[1]

        for i in range(5):
            logit = net0(x * 0.5**i)
            with torch.no_grad():
                target = net0(y).detach()
            loss += self.loss_fn(logit, target)
            logit = net1(x * 0.5**i)
            with torch.no_grad():
                target = net1(y).detach()
            loss += self.loss_fn(logit, target) * self.beta
        return None, loss


class TI_Ensemble_Attack(TI_Attack):
    def net_forward(self, x, y):
        loss = 0
        net0, net1 = self.net[0], self.net[1]
        logits = net0(x)
        with torch.no_grad():
            target = net0(y).detach()
        loss += self.loss_fn(logits, target)
        logits = net1(x)
        with torch.no_grad():
            target = net1(y).detach()
        loss += self.loss_fn(logits, target) * self.beta
        return None, loss


class DI_MI_Ensemble_Attack(DI_MI_Attack):
    def net_forward(self, x, y):
        loss = 0
        net0, net1 = self.net[0], self.net[1]
        logits = net0(x)
        with torch.no_grad():
            target = net0(y).detach()
        loss += self.loss_fn(logits, target)
        logits = net1(x)
        with torch.no_grad():
            target = net1(y).detach()
        loss += self.loss_fn(logits, target) * self.beta
        return None, loss


class ADMIX_Ensemble_Attack(ADMIX_Attack):
    def net_forward(self, x, y):
        loss = 0
        x = self.admix(x)
        net0, net1 = self.net[0], self.net[1]
        logits = net0(x)
        with torch.no_grad():
            target = net0(y).detach()
        loss += self.loss_fn(logits, target.repeat([self.repeat, 1]))
        logits = net1(x)
        with torch.no_grad():
            target = net1(y).detach()
        loss += self.loss_fn(logits, target.repeat([self.repeat, 1])) * self.beta
        return None, loss


class ADMIX_DI_MI_Ensemble_Attack(ADMIX_DI_MI_Attack):
    def net_forward(self, x, y):
        loss = 0
        x = self.admix(x)
        net0, net1 = self.net[0], self.net[1]
        logits = net0(x)
        with torch.no_grad():
            target = net0(y).detach()
        loss += self.loss_fn(logits, target.repeat([self.repeat, 1]))
        logits = net1(x)
        with torch.no_grad():
            target = net1(y).detach()
        loss += self.loss_fn(logits, target.repeat([self.repeat, 1])) * self.beta
        return None, loss


# class TIDIM_Ensemble_Attack(TIDIM_Attack):
#     def net_forward(self, x, y):
#         loss = 0
#         net0, net1 = self.net[0], self.net[1]
#         logits = net0(x)
#         target = net0(y)
#         loss += self.loss_fn(logits, target)
#         logits = net1(x)
#         target = net1(y)
#         loss += self.loss_fn(logits, target) * self.beta
#         return None, loss



# class TIDIM_Custom_Attack(TIDIM_Attack):
#     def net_forward(self, x, y):
#
#         loss = 0
#         r_net, n_net = self.net[0], self.net[1]
#         x_embd = r_net(x)
#         y_embd = r_net(y)
#         loss += self.loss_fn(x_embd, y_embd)
#
#         x_embd = n_net(x)
#         y_embd = n_net(y)
#         temp = self.loss_fn(x_embd, y) * 0.1
#         loss = loss - temp
#
#         x_embd = self.net(x)
#         loss = self.loss_fn(x_embd, y)
#
#         return None, loss

def select_attacker(args, model_generate, model_non=None, loss_fn=torch.nn.MSELoss(), random_init=True):

    if args.attack == 'DI':
        if args.mode == 'standard':
            attacker = DI_Attack(model_generate, loss_fn=loss_fn, random_init=random_init)
        else:
            attacker = DI_Ensemble_Attack([model_generate, model_non], loss_fn=loss_fn, random_init=random_init)
    elif args.attack == 'TI':
        if args.mode == 'standard':
            attacker = TI_Attack(model_generate, loss_fn=loss_fn, random_init=random_init)
        else:
            attacker = TI_Ensemble_Attack([model_generate, model_non], loss_fn=loss_fn, random_init=random_init)
    elif args.attack == 'MI' or args.attack == 'FGSM':
        if args.mode == 'standard':
            attacker = MI_Attack(model_generate, loss_fn=loss_fn, random_init=random_init)
        else:
            attacker = MI_Ensemble_Attack([model_generate, model_non], loss_fn=loss_fn, random_init=random_init)
    elif args.attack == 'TIDIM':
        if args.mode == 'standard':
            attacker = TIDIM_Attack(model_generate, loss_fn=loss_fn, random_init=random_init)
        else:
            attacker = None
    elif args.attack == 'SI':
        if args.mode == 'standard':
            attacker = SI_Attack(model_generate, loss_fn=loss_fn, random_init=random_init)
        else:
            attacker = SI_Ensemble_Attack([model_generate, model_non], loss_fn=loss_fn, random_init=random_init)
    elif args.attack == 'DI_MI':
        if args.mode == 'standard':
            attacker = DI_MI_Attack(model_generate, loss_fn=loss_fn, random_init=random_init)
        else:
            attacker = DI_MI_Ensemble_Attack([model_generate, model_non], loss_fn=loss_fn, random_init=random_init)
    elif args.attack == 'ADMIX':
        if args.mode == 'standard':
            attacker = ADMIX_Attack(model_generate, loss_fn=loss_fn, random_init=random_init)
        else:
            attacker = ADMIX_Ensemble_Attack([model_generate, model_non], loss_fn=loss_fn, random_init=random_init)
    elif args.attack == 'ADMIX_DI_MI':
        if args.mode == 'standard':
            attacker = ADMIX_DI_MI_Attack(model_generate, loss_fn=loss_fn, random_init=random_init)
        else:
            attacker = ADMIX_DI_MI_Ensemble_Attack([model_generate, model_non], loss_fn=loss_fn, random_init=random_init, beta=args.beta)
    else:
        attacker = None

    if attacker is not None:
        attacker.beta = args.beta
    return attacker
