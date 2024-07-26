import torch
import torch.optim as optim

def train_critic(critic, gen, real, fake, opt_critic, LAMBDA_GP, gradient_penalty):
    # Train Critic: max E[critic(real)] - E[critic(fake)]
    # equivalent to minimizing the negative of that
    critic_real = critic(real).reshape(-1)
    critic_fake = critic(fake).reshape(-1)
    gp = gradient_penalty(critic, real, fake)
    loss_critic = (
        -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
    )
    opt_critic.zero_grad()
    loss_critic.backward(retain_graph=True)
    opt_critic.step()
    return loss_critic

def train_generator(critic, gen, fake, opt_gen):
    # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
    gen_fake = critic(fake).reshape(-1)
    loss_gen = -torch.mean(gen_fake)
    opt_gen.zero_grad()
    loss_gen.backward()
    opt_gen.step()
    return loss_gen
