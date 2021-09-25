Stable-baselines implementation of PPO for SimFish. 

To work, requires following modifications to stable-baselines in file common/tf_layers.py: 
* to lstm() - add reuse=tf.AUTO_REUSE to tf.variable_scope()

* to linear() - add reuse=tf.AUTO_REUSE to tf.variable_scope()
