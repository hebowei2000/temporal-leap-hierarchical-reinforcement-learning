import tensorflow as tf
import tf.contrib.slim as slim   
from agents import ddpg_agent
BaseAgent = ddpg_agent.TD3Agent

""" What TD3 class has:
        action(): chooses action based on actor net
        sample_action(): chooses noisy action
        critic(): return Q value based on critic net
        value_net(): one-step way to evaluate both action and critic nets
        loss functions for both networks
        functions that return the network variables (W and b)
    
    What we need:
        additional network functions (hats) and loss



"""


class UpperAgent(BaseAgent):
    def __init__(self,
                observation_spec,
                action_spec,
                tf_env,
                tf_context,
                #step_cond_fn=cond_fn.env_transition,        # Do we need these?
                #reset_episode_cond_fn=cond_fn.env_restart,  #
                #reset_env_cond_fn=cond_fn.false_fn          #
                ):

        self.ACTOR_HAT_NET_SCOPE = 'actor_hat_net'
        self.CRITIC_HAT_NET_SCOPE = 'critic_hat_net'

        """Constructs a Meta agen   
        Args:
          observation_spec: A TensorSpec defining the observations.
          action_spec: A BoundedTensorSpec defining the actions.
          tf_env: A Tensorflow environment object.
          tf_context: A Context class.
          step_cond_fn: A function indicating whether to increment the num of steps.
          reset_episode_cond_fn: A function indicating whether to restart the
                episode, resampling the context.
          reset_env_cond_fn: A function indicating whether to perform a manual reset
                of the environment.
        Raises:
          ValueError: If 'dqda_clipping' is < 0.
        """
        self._actor_hat_net = tf.make_template(
            self.ACTOR_HAT_NET_SCOPE, self.actor_net, create_scope_now_=True)
        self._critic_hat_net = tf.make_template(
            self.CRITIC_HAT_NET_SCOPE, self.critic_net, create_scope_now_=True)

    def critic_hat_net(self, states, actions):
        """Returns the output of the critic hat network (Q')"""
        self._validate_states(states)
        self._validate_actions(actions)
        return self._critic_hat_net(states, actions)

    def best_next_states(self, states):
        """Returns the states that gives the highest Q value."""
        self._validate_states(states)
        return self._actor_hat_net(states, self._action_spec)
    
    def hat_value(self, states):
        """one-step computes the Q value using output from actor net."""
        actions = self.best_next_states(states)
        return self.critic_hat_net(states, actions)

    def critic_hat_loss(self, states, actions, rewards, discounts,
                  next_states):
        """Returns a 0-rank tensor representing the loss function for the critic net."""
        ys = rewards + discounts * self.hat_value(next_states)
        q_vals = self.critic_hat_net(states, actions)
        return self._td_errors_loss(ys, q_vals)
    
    def actor_hat_loss(self, states):
        """Returns a 0-rank tensor representing the loss function for the actor net."""
        self._validate_states(states)
        actions = self.actor_net(states, stop_gradients=False)
        critic_values = self.critic_net(states, actions)
        q_values = self.critic_function(critic_values, states)
        dqda = tf.gradients([q_values], [actions])[0]
        actions_norm = tf.norm(actions)
        actions_norm *= self._actions_regularizer
        return slim.losses.mean_squared_error(tf.stop_gradient(dqda + actions),
                                              actions,
                                              scope='actor_loss') + actions_norm


    