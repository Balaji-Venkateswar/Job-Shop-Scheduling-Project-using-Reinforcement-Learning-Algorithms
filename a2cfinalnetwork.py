import torch
import torch.nn as nn
import torch.nn.functional as F


class network(nn.Module):
    
    def __init__(self,inp_dim,hid_node_1,hid_node_2,n_act, n_out):
       
        super(network,self).__init__()
        
        self.actor_ly1 = nn.Linear(inp_dim,hid_node_1)
        self.a_ly1_ly2 = nn.Linear(hid_node_1,hid_node_2)
        self.actor_out = nn.Linear(hid_node_2,n_act)
        
        self.critic_ly1 = nn.Linear(inp_dim,hid_node_1)
        self.c_ly1_ly2 = nn.Linear(hid_node_1,hid_node_2)
        self.critic_out = nn.Linear(hid_node_2,n_out)
       
    def actorforward(self,state):
        state = state.float().unsqueeze(0)
        actor_ly1_relu = F.relu(self.actor_ly1(state))
        second_ly_relu = F.relu(self.a_ly1_ly2(actor_ly1_relu))
        prob_dist = F.softmax(self.actor_out(second_ly_relu), dim = 1)
        return prob_dist
           
    def criticforward(self,state):
        state = state.float().unsqueeze(0)
        
        critic_ly1_relu = F.relu(self.critic_ly1(state))
        second_critic_relu = F.relu(self.c_ly1_ly2(critic_ly1_relu))
        state_val = self.critic_out(second_critic_relu)
        
        return state_val
    

    
class Actor():
    
    def __init__(self, gamma):    
        self.gamma = gamma
        
    def choose_action(self,prob):
        action = torch.multinomial(prob, 1)
        return action
 
    def actorlearn(self, advantage, action, prob):
        log_prob = torch.log(prob.squeeze(0)[action])
        actorloss = -log_prob * advantage
        return actorloss
    
    
class Critic():
    def __init__(self,gamma):
        self.gamma = gamma
       
        
    def criticslearn(self, reward, state_val, nxt_state_val):
        advan_fn = reward + (self.gamma * nxt_state_val) - state_val
        critic_loss = 0.5 * advan_fn ** 2
        return critic_loss, advan_fn

