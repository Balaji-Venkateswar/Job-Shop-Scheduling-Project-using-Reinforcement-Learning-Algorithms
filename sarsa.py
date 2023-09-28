from JSSP_Environment import env
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import math


file = r'sasd_a2c.xlsx'
load_sp = pd.ExcelFile(file)

# state_action_state' table
df1 = load_sp.parse('state_action_nstate_reward')
state_action_nstate = torch.FloatTensor(df1.values)

# action index
df2 = load_sp.parse('action_index')
action_index_data = torch.FloatTensor(df2.values)


e = 0.9
gamma = 0.9                           
alpha = 0.01                   
episodes = 2000
state_1 = 0
q_table=torch.zeros(27,100)

'''selecting action value'''

def action(state):
    prob_action = np.random.uniform(0,1)
    action_index = action_index_data[:,0]
    if prob_action > e: 
        action_selected = np.random.choice(action_index)
        
    else:
        q_max = np.argmax(q_table[int(state)])
        action_selected = q_max
        
    return int(action_selected)
    
'''updation of Q-value'''

def q_val_upd(state1, action1, state2, action2, reward):
    q_old = q_table[state1,action1]
    target = reward + gamma * (q_table[state2,action2])
    q_new = q_old + alpha * (target - q_old)
    q_table[state1,action1] = q_new 
    
''' to count output'''
        
def output_count(a):
    count1=0
    count2=0
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
    
    
if __name__ == "__main__":
    tot_rew, tot_q, tot_steps, tot_goal = [], [], [], []
    tot_op1, tot_op2 = [], []
     
    for epi_no in range(episodes):

        print(epi_no)
        envir = env()
        state_1, step_epi = 0, 0
        rew_epi = []
        q_epi = []
        op1_epi, op2_epi = [], []
        action_1 = action(state_1)
        while True:
            state_2,reward,t_f = envir.next_state(action_1)
            rew_epi.append(reward)
            op1, op2 = output_count(action_1)
            op1_epi.append(op1)
            op2_epi.append(op2)
            action_2 = action(state_2)
            op1, op2 = output_count(action_2)
            op1_epi.append(op1)
            op2_epi.append(op2)
            q_val_upd(int(state_1), action_1, int(state_2), action_2, reward)
            state_1 = state_2
            action_1 = action_2
            step_epi += 1
            
            
            if t_f==True:
                break
        if reward == 50:
            goal_count = 1
        else:
            goal_count = 0
               
        a = np.sum(op1_epi)
        b = np.sum(op2_epi)
        tot_op1.append(a)
        tot_op2.append(b)
        np_rew = np.sum(rew_epi)
        tot_rew.append(np_rew)
        q_mean = torch.mean(q_table)
        tot_q.append(q_mean)
        tot_steps.append(step_epi)
        tot_goal.append(goal_count)
 
         
 
    q_excel1 = pd.DataFrame(tot_q) 
    q_excel1.to_excel(excel_writer="Q-values(e=0.9).xlsx")
    q_excel2 = pd.DataFrame(q_table)
    q_excel2.to_excel(excel_writer="Q-table.xlsx")
    q_excel3 = pd.DataFrame(tot_rew)
    q_excel3.to_excel(excel_writer="rew.xlsx")
    q_excel4 = pd.DataFrame(tot_steps)
    q_excel4.to_excel(excel_writer="step.xlsx")
    q_excel5 = pd.DataFrame(tot_op1)
    q_excel5.to_excel(excel_writer="op1.xlsx")
    q_excel6 = pd.DataFrame(tot_op2)
    q_excel6.to_excel(excel_writer="op2.xlsx")

    x1 = np.arange(episodes)      # Graph plots
    plt.plot(x1,tot_rew)
    plt.xlabel('No. of episodes')
    plt.ylabel('Sum of rewards per episode')
    plt.title('Rewards vs Episodes [epsilon = 0.9, gamma = 0.9, alpha = 0.01]')
    plt.grid(True)
    plt.savefig("rew.pdf")
    plt.show()
    
    x2 = np.arange(episodes)
    plt.plot(x2,tot_q)
    plt.xlabel('No. of episodes')
    plt.ylabel('q-values')
    plt.title(' Mean q-values vs Episodes [eps = 0.9, gamma = 0.9, alpha = 0.01]')
    plt.grid(True)
    plt.savefig("q-val.pdf")
    plt.show()
    
    x3 = np.arange(episodes)
    plt.plot(x3,tot_goal)
    plt.xlabel('No. of episodes')
    plt.ylabel('Goal')
    plt.title(' Goal vs Episodes [epsilon = 0.9, gamma = 0.9, alpha = 0.01]')
    plt.grid(True)
    plt.savefig("goal.pdf")
    plt.show()
    
    x4 = np.arange(episodes)
    plt.plot(x4,tot_steps)
    plt.xlabel('No. of episodes')
    plt.ylabel('Steps')
    plt.title(' Steps per episode [eps = 0.9, gamma = 0.9, alpha = 0.01]')
    plt.grid(True)
    plt.savefig("step.pdf")
    plt.show()
           
    x5 = np.arange(episodes)
    plt.subplot(2,1,1)
    plt.plot(x5,tot_op1,'o-', color = "C2")
    plt.ylabel('output 1')
    plt.title(' Outputs per episode [eps = 0.9, gamma = 0.9, alpha = 0.01]')
    plt.grid(True)

    x6 = np.arange(episodes)
    plt.subplot(2,1,2)
    plt.plot(x6,tot_op2,'.-',color = "C1")
    plt.xlabel('No. of episodes')
    plt.ylabel('output 2')
    plt.grid(True)
    plt.savefig("output.pdf")
    plt.show()
    
        
        
        
        
        
        
        
        
        
        
    