import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
import matplotlib.pyplot as plt

# 常数定义
hbar = 197.3269804
m = 939
omega = 10 / hbar

# Swish激活函数
def swish(x):
    return x * nn.sigmoid(x)

# 神经网络定义
class RealNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x: [batch, 3]
        h = nn.Dense(128)(x)
        h = swish(h)
        h = nn.Dense(128)(h)
        h = swish(h)
        h = nn.Dense(64)(h)
        h = swish(h)
        # 输出实数波函数
        psi = nn.Dense(1)(h)[..., 0]
        return psi

class RealNet2(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x: [batch, 3]
        h = nn.Dense(128)(x)
        h = swish(h)
        h = nn.Dense(128)(h)
        h = swish(h)
        h = nn.Dense(64)(h)
        h = swish(h)
        # 输出实数波函数
        psi = nn.Dense(1)(h)[..., 0]
        return psi

# 生成输入数据
x = jnp.arange(-20, 20.01, 0.02)
inputs = jnp.stack([x, jnp.cos(x), jnp.sin(x)], axis=-1)

# 损失函数
def loss_fn(params, model, x, inputs, other_params=None, other_model=None):
    # 计算波函数值
    psi = jax.vmap(lambda inp: model.apply(params, inp), in_axes=0)(inputs)
    
    # 计算对x的二阶导数
    def psi_wrt_x(xi):
        # 创建输入，只改变x值，保持cos(x)和sin(x)不变
        inp = jnp.array([xi, jnp.cos(xi), jnp.sin(xi)])
        return model.apply(params, inp[None, :])[0]  # 添加batch维度并取第一个元素
    
    # 计算二阶导数
    psi_2nd = jax.vmap(lambda xi: jax.grad(jax.grad(psi_wrt_x))(xi))(x)

    # Hamiltonian作用
    kinetic = - (hbar ** 2) / (2 * m) * psi_2nd
    potential = 0.5 * m * (omega ** 2) * x * x * psi
    H_psi = kinetic + potential

    # 分子: <psi|H|psi>
    numerator = jnp.trapezoid(psi * H_psi, x)
    # 分母: <psi|psi>
    denominator = jnp.trapezoid(psi * psi, x)
    energy = jnp.real(numerator / denominator)
    
    # 添加正交项
    if other_params is not None and other_model is not None:
        psi_other = jax.vmap(lambda inp: other_model.apply(other_params, inp), in_axes=0)(inputs)
        # 归一化其他波函数
        norm_other = jnp.sqrt(jnp.trapezoid(psi_other ** 2, x))
        psi_other_norm = psi_other / norm_other
        # 计算重叠积分
        overlap = jnp.trapezoid(psi * psi_other_norm, x)
        # 正交惩罚项
        orthogonality_penalty = 10.0 * overlap ** 2
        loss = energy + orthogonality_penalty
    else:
        loss = energy
    
    return loss

# 初始化模型和优化器
model1 = RealNet()
model2 = RealNet2()
rng = jax.random.PRNGKey(0)
params1 = model1.init(rng, inputs)
params2 = model2.init(rng, inputs)
tx = optax.adam(learning_rate=1e-3)
state1 = train_state.TrainState.create(apply_fn=model1.apply, params=params1, tx=tx)
state2 = train_state.TrainState.create(apply_fn=model2.apply, params=params2, tx=tx)

# 训练循环
@jax.jit
def train_step1(state1, x, inputs):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state1.params, model1, x, inputs)
    state1 = state1.apply_gradients(grads=grads)
    return state1, loss

@jax.jit
def train_step2(state1, state2, x, inputs):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state2.params, model2, x, inputs, state1.params, model1)
    state2 = state2.apply_gradients(grads=grads)
    return state2, loss

# 先训练第一个网络
num_epochs1 = 2000
losses1 = []
print("训练第一个网络...")
for epoch in range(num_epochs1):
    state1, loss1 = train_step1(state1, x, inputs)
    losses1.append(loss1)
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss1: {loss1}")

# 再训练第二个网络
num_epochs2 = 2000
losses2 = []
print("训练第二个网络...")
for epoch in range(num_epochs2):
    state2, loss2 = train_step2(state1, state2, x, inputs)
    losses2.append(loss2)
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss2: {loss2}")

# 得到最终psi并归一化
def get_psi(params, model, x, inputs):
    psi = jax.vmap(lambda inp: model.apply(params, inp), in_axes=0)(inputs)
    norm = jnp.sqrt(jnp.trapezoid(psi ** 2, x))
    psi_norm = psi / norm
    return psi_norm

psi_pred1 = get_psi(state1.params, model1, x, inputs)
psi_pred2 = get_psi(state2.params, model2, x, inputs)

# 计算解析解
def analytical_ground_state(x):
    # 基态: psi_0(x) = (m*omega/(pi*hbar))^(1/4) * exp(-m*omega*x^2/(2*hbar))
    alpha = jnp.sqrt(m * omega / hbar)
    return (alpha / jnp.sqrt(jnp.pi)) ** 0.5 * jnp.exp(-alpha**2 * x**2 / 2)

def analytical_first_excited(x):
    # 第一激发态: psi_1(x) = sqrt(2) * (m*omega/(pi*hbar))^(1/4) * x * exp(-m*omega*x^2/(2*hbar))
    alpha = jnp.sqrt(m * omega / hbar)
    return jnp.sqrt(2) * (alpha / jnp.sqrt(jnp.pi)) ** 0.5 * alpha * x * jnp.exp(-alpha**2 * x**2 / 2)

# 计算解析解
psi_analytical_ground = analytical_ground_state(x)
psi_analytical_excited = analytical_first_excited(x)

# 调整拟合结果符号使其与解析解一致
overlap_ground = jnp.trapezoid(psi_pred1 * psi_analytical_ground, x)
overlap_excited = jnp.trapezoid(psi_pred2 * psi_analytical_excited, x)

# 根据重叠积分的符号调整拟合结果
if overlap_ground < 0:
    psi_pred1 = -psi_pred1
if overlap_excited < 0:
    psi_pred2 = -psi_pred2

# 可视化第一个网络
plt.figure(figsize=(10,5))
plt.plot(np.array(x), np.array(psi_pred1), label='Neural network fitting results', color='blue', linewidth=2)
plt.plot(np.array(x), np.array(psi_analytical_ground), label='Analytical solution', color='red', linestyle='--', linewidth=2)
plt.title('Neural network fitting of the ground state of one-dimensional oscillator')
plt.xlabel('x')
plt.ylabel('psi')
plt.legend()
plt.grid(True)
plt.show()

# 可视化第二个网络
plt.figure(figsize=(10,5))
plt.plot(np.array(x), np.array(psi_pred2), label='Neural network fitting results', color='blue', linewidth=2)
plt.plot(np.array(x), np.array(psi_analytical_excited), label='Analytical solution', color='red', linestyle='--', linewidth=2)
plt.title('Neural network fitting of excited states of one-dimensional oscillator')
plt.xlabel('x')
plt.ylabel('psi')
plt.legend()
plt.grid(True)
plt.show()

# 计算重叠积分验证正交性
overlap = jnp.trapezoid(psi_pred1 * psi_pred2, x)
print(f"重叠积分: {overlap}")

# 绘制第一个网络的损失函数
plt.figure(figsize=(10,5))
plt.plot(losses1, label='energy', color='blue')
plt.title('Energy for neural network fitting of the ground state of one-dimensional oscillator')
plt.xlabel('Steps')
plt.ylabel('Energy')
plt.legend()
plt.grid(True)
plt.show()

# 绘制第二个网络的损失函数
plt.figure(figsize=(10,5))
plt.plot(losses2, label='energy', color='red')
plt.title('Energy for neural network fitting of excited states of one-dimensional oscillator')
plt.xlabel('Steps')
plt.ylabel('Energy')
plt.legend()
plt.grid(True)
plt.show()

# 计算最后500步的平均值
if len(losses1) >= 500:
    avg_loss1 = np.mean(losses1[-500:])
    print(f"第一个网络最后500步平均能量: {avg_loss1}")
else:
    print(f"第一个网络训练步数不足500步，实际步数: {len(losses1)}")

if len(losses2) >= 500:
    avg_loss2 = np.mean(losses2[-500:])
    print(f"第二个网络最后500步平均能量: {avg_loss2}")
else:
    print(f"第二个网络训练步数不足500步，实际步数: {len(losses2)}")
