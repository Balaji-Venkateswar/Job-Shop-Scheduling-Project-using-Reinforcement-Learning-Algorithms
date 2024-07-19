import torch
import numpy as np
import pandas as pd
import math

'''
Ensure that the excel file 'sasd_a2c.xlsx' is always kept in the same folder as the 'JSSP_Environment.py'
'''

file = r'sasd_a2c.xlsx'
load_sp = pd.ExcelFile(file)

# state_action_state' table
df1 = load_sp.parse('state_action_nstate_reward')
state_action_nstate = torch.FloatTensor(df1.values)

# action index
df2 = load_sp.parse('action_index')
action_index_data = torch.FloatTensor(df2.values)

class env():

    def __init__(self):
        super(env, self).__init__()
        self.s = 0
        self.count_op1 = 0
        self.count_op2 = 0
        self.count_ip1 = 0
        self.count_ip2 = 0
        self.done = False

    def check(self, n_state, rew):
        '''Method to reset the environment based on terminal conditions of States'''
        if n_state == 14 or n_state == 17:
            rew = torch.tensor([-50])
            self.done = True
        return n_state, rew

    def check1(self, count_op1, count_op2,rew):
        '''Method to reset the environment based on terminal conditions of Output counts'''
        if self.count_op1 >= 10 and self.count_op2 >= 10:
            rew = torch.tensor([+50])
            self.done = True
        return rew

    def check2(self, count_ip1, count_ip2,rew):
        '''Method to reset the environment based on terminal conditions of Input counts'''
        if self.count_ip1 >= 15:
            rew = torch.tensor([-50])
            self.done = True

        elif self.count_ip2 >= 15:
            rew = torch.tensor([-50])
            self.done = True
        return rew

    def reset(self):
        '''Method to reset the Environment'''
        self.s = torch.zeros(1)
        self.count_op1 = 0
        self.count_op2 = 0
        self.count_ip1 = 0
        self.count_ip2 = 0



    def next_state(self, a):

        '''Here in this method we give action "a" as input
        and receive state "s", reward "rew", info of reset as outputs'''

        a1 = math.trunc(a/10)
        a2 = a - (a1*10)
        y = state_action_nstate[np.where(state_action_nstate[:,0]==self.s),:]
        z = y[0][np.where(y[0][:,1]==a1),:]
        self.s = z[0][np.where(z[0][:,2]==a2),3].view(-1)
        op1_count = z[0][np.where(z[0][:,2]==a2),4].view(-1)
        op2_count = z[0][np.where(z[0][:,2]==a2),5].view(-1)
        rew1 = z[0][np.where(z[0][:,2]==a2),6].view(-1)
        rew2 = z[0][np.where(z[0][:,2]==a2),7].view(-1)
        ip1_count = z[0][np.where(z[0][:,2]==a2),8].view(-1)
        ip2_count = z[0][np.where(z[0][:,2]==a2),9].view(-1)
        self.count_op1+=op1_count
        self.count_op2+=op2_count
        self.count_ip1+=ip1_count
        self.count_ip2+=ip2_count
        rew = rew1 + rew2

        self.s, rew= self.check(self.s, rew)
        rew = self.check1(self.count_op1, self.count_op2, rew)
        rew = self.check2(self.count_ip1, self.count_ip2, rew)
        if self.done == True:
            self.reset()

        return self.s, rew, self.done

