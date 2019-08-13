import tensorflow as tf
import tf.contrib.slim as slim
from agents import ddpg_agent
BaseAgent = ddpg_agent.TD3Agent


class LowerAgent(BaseAgent):
    def __init__(self,
                observation_spec,
                action_spec,
                tf_env,
                tf_context,
                #step_cond_fn=cond_fn.env_transition,        # Do we need these?
                #reset_episode_cond_fn=cond_fn.env_restart,  #
                #reset_env_cond_fn=cond_fn.false_fn          #
                ):

        def set_meta_agent(self, agent):
            self.meta_agent = agent
        
        @property
        def meta_agent(self):
            return self.meta_agent
            

    