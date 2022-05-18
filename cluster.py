"""
Groups of servers example

Covers:

- Resources: Resource
- Resources: Container
- Waiting for other processes

Scenario:
  A cluster has a number of hosts. Each host has a number of servers. 
  Customers randomly arrive at the cluster, request one
 server  and obtain service from that server.

  A cluster control process manages the resource (switch on and off the host).

"""
import ray
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.env.external_env import ExternalEnv
#from ray.rllib.evaluation.tests.test_rollout_worker import BadPolicy, MockPolicy
#from ray.rllib.examples.env.mock_env import MockEnv
from ray.rllib.utils.test_utils import framework_iterator
from ray.tune.registry import register_env

import gym
import numpy as np

import sys


import simpy
import random



from tdigest import TDigest
from enum import IntEnum
import itertools



RANDOM_SEED = 42

BOOT_TIME = 5      # Seconds it takes the tank truck to arrive
SIM_TIME = 55000            # Simulation time in seconds

CONTROL_TIME_INTERVAL=30.0
NUMBER_OF_HOSTS=10
NUMBER_OF_SERVERS=5
ARRIVAL_RATE = 20.
SERVICE_RATE = 2.0
SERVICE_TIME=1.2
RESPONSE_TIME_THRESHOLD_D= 1.3
RESPONSE_TIME_THRESHOLD_U=4.0


class Action(IntEnum):
    ScaleOut   = 0
    Do_nothing = 1
    ScaleIn    = 2
    
    

class PowerState(IntEnum):
    OFF =0
    ON_A  =1 # ON and accept customers
    ON_N  =2 # ON and does not accept customers
    
    
class Cluster(object):
    """A queue has a limited number of servers (``NUM_SERVERS``) to
    serve customers in parallel.

    customers have to request one of the servers. When they got one, they
    can start the serving processes and wait for it to finish (which
    takes ``washtime`` minutes).

    """
    def __init__(self,env,number_of_hosts, number_of_servers):
        self.env = env
        self.number_of_hosts=number_of_hosts
        self.number_of_servers=number_of_servers
        self.hosts={i: simpy.Resource(self.env,self.number_of_servers) for i in range(self.number_of_hosts)}
        #self.host_state={i: 0 for i in range(self.number_of_hosts)}
        self.host_state=np.array([0 for _ in  range(self.number_of_hosts)])
        self.host_state[0]=PowerState.ON_A # 1: on, 0: off, 3: on -- do not allocate
        self.digest=TDigest() # response time
        self.arrdigest= TDigest() # arrival time
        self.active_num=1
        self.arrdigest.update(100.)
        self.digest.update(0.)

#        self.buffer=self.env.Store()
    def search_for_allocation(self):
        ihostwith_smallest=self.number_of_hosts-1
        for j in range(self.number_of_hosts):
           if self.host_state[j]==PowerState.ON_A:
                ihostwith_smallest=j
                break
        for i in range(j+1,self.number_of_hosts):
           if self.host_state[i]==PowerState.ON_A and len(self.hosts[i].queue)+self.hosts[i].count < len(self.hosts[ihostwith_smallest].queue)+self.hosts[ihostwith_smallest].count:
                ihostwith_smallest=i
        # print(self.number_of_hosts) 
        assert ihostwith_smallest < self.number_of_hosts, f"number small than {self.number_of_hosts} expected, got: {ihostwith_smallest}"
        return ihostwith_smallest
    def search_for_off_host(self):
        hostid=self.number_of_hosts
        for j in range(self.number_of_hosts):
           if self.host_state[j]==PowerState.ON_N:
             hostid=j
             #print("Terminate the switch off")
             break
        if hostid== self.number_of_hosts:
          for i in range(self.number_of_hosts):
             if self.host_state[i]==PowerState.OFF:
               hostid=i  
               break
        return hostid
    
def customer(name, env, cluster):
    """A customer arrives at the cluster for service.
    It requests one of the servers. If there is no server available, 
    the customer has to wait for a new server (takes time to boot a host.
    """
    #print('No of active hosts %d' % cluster.active_num)
    #print('%s arriving at cluster at %.1f' % (name, env.now))
    assignedto=cluster.search_for_allocation()
    #print("Host %d queue size: %d " % (assignedto,len(cluster.hosts[assignedto].queue)))
    #print("Being served:", cluster.hosts[assignedto].count)
    with cluster.hosts[assignedto].request() as req:
        start = env.now
        # Request one of the servers from hosts[idh]
        yield req                
        # The "actual" service process takes some time
        t=random.expovariate(SERVICE_RATE)
        #t=random.uniform(0.01,2)
        #t=1
        yield env.timeout(t)
        if cluster.hosts[assignedto].count==0 and cluster.host_state[assignedto]==PowerState.ON_N:
          cluster.host_state[assignedto]=PowerState.OFF
          cluster.active_num -=1
        cluster.digest.update(env.now - start)
        #print('No of active hosts %d' % cluster.active_num)
        #print('%s finished service in %.7f seconds. service time %.7f' % (name,
        #                                                     env.now - start,t))





def start_host(env,cluster,hostid):
    yield env.timeout(BOOT_TIME)
    #print('Host %d ready at time %d' % (hostid,env.now))
    #print('No of active hosts %d' % cluster.active_num)
    cluster.host_state[hostid]=PowerState.ON_A
    cluster.active_num +=1
    

def customer_generator(env, cluster):
    """Generate new customers that arrive at the cluster."""
    for i in itertools.count():
        if env.now <SIM_TIME/2:
           t=random.expovariate(ARRIVAL_RATE)
        elif env.now <SIM_TIME*3/4:
           t=random.expovariate(ARRIVAL_RATE/2)
        else:
           t=random.expovariate(ARRIVAL_RATE/4)
        cluster.arrdigest.update(t)
        yield env.timeout(t)
        env.process(customer('Customer %d' % i, env, cluster,))


class ClusterEnv(ExternalEnv):
    def __init__(self,number_of_hosts,num_servers,percentile_points):

        self.number_of_hosts=number_of_hosts
        self.num_servers=num_servers
        self.percentile_points=percentile_points
        self.number_of_active_hosts=1
        self.observation_space = gym.spaces.Tuple(
          [
             gym.spaces.Box(0,np.inf,shape=(self.percentile_points,),dtype=np.float64),  # Arrival cdf
             gym.spaces.Box(0,np.inf,shape=(self.percentile_points,),dtype=np.float64), # response
             gym.spaces.Box(0,2,shape=(self.number_of_hosts,),dtype=np.int64),      # state of hosts -- off, ON_A, ON_N 
             gym.spaces.Box(0,10000,shape=(self.number_of_hosts,),dtype=np.int64), # number of customers at hosts
             #Discrete(self.number_of_hosts), #number of active hosts,  number of servers
             #Discrete(self.number_of_host*self.capacity+1)
          ]
        )
        print("Initialization ---")
        #print(self.observation_space)
        print("Initialization ---")
        self.action_space = gym.spaces.Discrete(3)
        ExternalEnv.__init__(self, self.action_space, self.observation_space)
        self.simenv = simpy.Environment()
        self.cluster = Cluster(self.simenv,self.number_of_hosts,self.num_servers)
        self.arr=np.zeros(shape=(self.percentile_points,))
        self.ser=np.zeros(shape=(self.percentile_points,))
        self.numofcustomer= np.zeros(shape=(self.cluster.number_of_hosts,))
        for i in range(self.cluster.number_of_hosts):
            self.numofcustomer[i] = len(self.cluster.hosts[i].queue)+self.cluster.hosts[i].count
            
    def step(self,action):
        print("STEP?")
        return 100,100, 100,{} 
    
    def cluster_control(self):
    #"""Periodically manages the cluster"""
        #print("cluster control ---")
        yield self.simenv.timeout(CONTROL_TIME_INTERVAL)
        for i in range(self.percentile_points):
            self.arr[i]=self.cluster.arrdigest.percentile(i)
            self.ser[i]=self.cluster.digest.percentile(i)           
        for i in range(self.cluster.number_of_hosts):
            self.numofcustomer[i] = len(self.cluster.hosts[i].queue)+self.cluster.hosts[i].count
          
        obs=tuple([self.arr,self.ser,self.cluster.host_state,self.numofcustomer])
        while True:
            # perform acion
            #
            #print("cluster control ")
            self.eid= self.start_episode()
            #print(self.eid)
            
            self.action = self.action_space.sample()
            #print(self.action)
            #print("Obs: ", obs)
            if self.action==Action.ScaleIn and self.cluster.active_num==1:
                self.action==Action.Do_nothing
            
            if self.action==Action.ScaleOut and self.cluster.active_num==self.cluster.number_of_hosts:
                self.action==Action.Do_nothing

            self.log_action(self.eid, obs, self.action)
            #print("Action %d" % self.action)
            #print(obs)
            
            #action = self.get_action(eid, obs)
            if self.action== Action.ScaleOut:
                hostid=self.cluster.search_for_off_host()
                #print("scale out, host %d" %hostid)
                if hostid<self.cluster.number_of_hosts: 
                    if self.cluster.host_state[hostid]==PowerState.OFF:
                        yield self.simenv.process(start_host(self.simenv,self.cluster,hostid))
                    else:    
                        self.cluster.host_state[hostid]=PowerState.ON_A
            elif self.action==Action.ScaleIn:
                hostid=self.cluster.search_for_allocation()
                #print("scale in, host %d" % hostid)
                if self.cluster.hosts[hostid].count==0 and len(self.cluster.hosts[hostid].queue)==0 :
                    self.cluster.host_state[hostid]=PowerState.OFF
                    self.cluster.active_num -=1
                else:
                    self.cluster.host_state[hostid]=PowerState.ON_N
                    #print('No of active hosts %d' % self.cluster.active_num)
            # Check every 10 seconds
            # reward here and observation
            del self.cluster.digest
            self.cluster.digest=TDigest()
            del self.cluster.arrdigest
            self.cluster.arrdigest= TDigest()
            
            yield self.simenv.timeout(CONTROL_TIME_INTERVAL)
            
            for i in range(self.percentile_points):
                self.arr[i]=self.cluster.arrdigest.percentile(i)
                self.ser[i]=self.cluster.digest.percentile(i)
           
            for i in range(self.cluster.number_of_hosts):
                self.numofcustomer[i]= len(self.cluster.hosts[i].queue)+self.cluster.hosts[i].count
                if self.numofcustomer[i]<0:
                    print("ERROR")
            obs=tuple([self.arr,self.ser,self.cluster.host_state,self.numofcustomer])
            
            if self.cluster.digest.percentile(99.9) >RESPONSE_TIME_THRESHOLD_U:
                reward= -1.0* (self.action+1)
            else:
                reward= self.action+1
            #compute reward, obs,info
            info=""
            #print(obs)
            #print(reward)
            self.log_returns(self.eid, reward, info=info)
            self.end_episode(self.eid, obs)
             
            del self.cluster.digest
            self.cluster.digest=TDigest()
            del self.cluster.arrdigest
            self.cluster.arrdigest= TDigest()
            #print('No of active hosts %d' % cluster.active_num)
            #print("99.9 percentile %f" % cluster.digest.percentile(99.9))
            #print("60 percentile %f" % cluster.digest.percentile(60))
            #print("50 percentile %f" % cluster.digest.percentile(50))
            #print("10 percentile %f" % cluster.digest.percentile(10))
            # a consequence of  the previous action
            # construct observation
            
       

    def run(self):
        #print("Start -------")
        #random.seed(RANDOM_SEED)
        
        
        #np.array([(len(self.cluster.hosts[i].queue)+self.cluster.hosts[i].count for i in range(self.cluster.number_of_hosts))])
        # self.hoststate=    numpy.array([self.cluster.host_state[i] for i in range(self.cluster.number_of_hosts)]) 
        self.simenv.process(self.cluster_control())
        
        self.simenv.process(customer_generator(self.simenv, self.cluster))
        print("Start sim...")
        self.simenv.run(until=SIM_TIME)
        print("sim vege...")

if __name__ == "__main__":

        ray.init()

        random.seed(RANDOM_SEED)

        register_env(
            "Multiserver-v1",
            lambda _: ClusterEnv(NUMBER_OF_HOSTS,NUMBER_OF_SERVERS,100),
        )
        config = {
            "num_workers": 8,
            "disable_env_checking": True,
            #"checkpoint_at_end": True,
            "framework": "tf",
            "rollout_fragment_length":20,
            "exploration_config": {"epsilon_timesteps": 100},
        }
        #checkpoint_at_end=True
        #for _ in framework_iterator(config, frameworks=("tf2", "torch")):
        dqn = DQNTrainer(env="Multiserver-v1", config=config)
        reached = False
        for i in range(8):
                result = dqn.train()
                #if i % 3 == 0:
                checkpoint = dqn.save()
                print("checkpoint saved at", checkpoint)
                print(
                    "Iteration {}, reward {}, timesteps {}".format(
                        i, result["episode_reward_mean"], result["timesteps_total"]
                ))
                #if result["episode_reward_mean"] >= 80:
                #    reached = True
                #    break
            #if not reached:
             #   raise Exception("failed to improve reward")

        ray.shutdown()
        sys.exit()
