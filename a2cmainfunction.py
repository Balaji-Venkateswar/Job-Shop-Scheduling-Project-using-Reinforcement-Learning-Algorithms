from JSSP_Environment import env
import numpy as np
import torch
import torch.autograd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from a2cfinalnetwork import Actor, Critic, network
import math
import pandas as pd


def output_count(a):
    count1 = 0
    count2 = 0
    a1 = math.trunc(a/10)
    a2 = a - (a1*10)          
    if a1 == 4 or a1 == 8:
        count1 += 1
    else:
        count1 = 0    
    if a2 == 4 or a2 == 8:
        count2 += 1
    else:
        count2 = 0
    return count1,count2



no_inputs = 27
hid_node1 = 5
hid_node2 = 5
no_outputs = 100
episodes = 2000
gamma = 0.9
n_out = 1

actor = Actor(gamma)
critic = Critic(gamma)

a2c = network(no_inputs, hid_node1, hid_node2, no_outputs, n_out)

optimizer = optim.Adam(a2c.parameters(), lr = 0.0003)


tot_steps = []
tot_rew_epi = []
actcritloss = []
wp1count, wp2count = [], []

for episode in range(episodes):
    print(episode)
    state_s = torch.tensor(0)
    f = env()
    steps = 0
    reward, countop1, countop2 = 0, 0, 0
    a = 0

    while True:
        
        state = F.one_hot(state_s, 27) 
        prob = a2c.actorforward(state)
        action = actor.choose_action(prob)
        nxt_state,r,done = f.next_state(int(action))
        state_val = a2c.criticforward(state)
        state_s = torch.tensor(int(nxt_state))
        nxt_state_t = F.one_hot(state_s, 27)
        nxt_state_val = a2c.criticforward(nxt_state_t)
        criticloss, advantage = critic.criticslearn(r, state_val, nxt_state_val)
        actorloss = actor.actorlearn(advantage, action, prob) 
        a2closs = actorloss + criticloss

        a += a2closs
        
        output1, output2 = output_count(int(action))
        countop1 += output1
        countop2 += output2
          
        reward += r
        steps += 1    
        if done == True:
            break

    actcritloss.append(a.item()) 
    
    optimizer.zero_grad()

    a.backward()

    optimizer.step()


    wp1count.append(countop1)
    wp2count.append(countop2)
    tot_rew_epi.append(reward)
    tot_steps.append(steps)
    
    
epi = np.arange(episodes)
plt.plot(epi,tot_rew_epi)
plt.xlabel('episodes(nos)')
plt.ylabel('reward')
plt.title('Rewards obtained in each episode lr-0.0005, epi-1500, gamma - 0.9')
plt.grid(True)
plt.savefig("a2crewards5.pdf")
plt.show()


plt.plot(epi,tot_steps)
plt.xlabel('episodes(nos)')
plt.ylabel('steps')
plt.title('Total steps in each episode lr-0.0005, epi-1500, gamma - 0.9')
plt.grid(True)
plt.savefig("a2csteps5.pdf")
plt.show()


plt.plot(epi,actcritloss)
plt.xlabel('episodes(nos)')
plt.ylabel('loss')
plt.title('a2c loss per episode lr-0.0005, epi-1500, gamma - 0.9')
plt.grid(True)
plt.savefig("a2closs5.pdf")
plt.show()


plt.subplot(2,1,1)
plt.plot(epi,wp1count, label = 'WP1', color = "C1")
plt.ylabel('output1')
plt.title('output counts- lr-0.0005, epi-1500, gamma - 0.9')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(epi,wp2count, label = 'WP2', color = "C3")
plt.xlabel('episodes(nos)')
plt.ylabel('output2')
plt.grid(True)
plt.savefig("a2coutputs5.pdf")
plt.show()

tot_rew_epi = np.array(tot_rew_epi)
excel1 = pd.DataFrame(tot_rew_epi) 
excel1.to_excel(excel_writer="a2crewards.xlsx")

excel2 = pd.DataFrame(tot_steps) 
excel2.to_excel(excel_writer="a2csteps.xlsx")

wp1count = np.array(wp1count)
excel3 = pd.DataFrame(wp1count) 
excel3.to_excel(excel_writer="a2c w1output.xlsx")

wp2count = np.array(wp2count)
excel4 = pd.DataFrame(wp2count) 
excel4.to_excel(excel_writer="a2c w2output.xlsx")