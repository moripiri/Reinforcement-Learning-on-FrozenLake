# for i in range(len(trajectory)):
#     with tf.GradientTape() as tape:
#         G = reduce(lambda acc, x: acc + (self.gamma ** x[0]) * x[1], enumerate(reward_list[i + 1:]), 0)
#
#         s = np.eye(self.state_dim)[trajectory[i]['s']]  # one-hot encoding
#         s = np.expand_dims(s, axis=0)  # expand state dimension
#
#         policy = self.policy(s)
#         dist = tfp.distributions.Categorical(probs=policy)
#         log_policy = tf.reshape(dist.log_prob(trajectory[i]['a']), (-1, 1))
#
#         loss = -(self.gamma ** i) * G * log_policy
#
#     variables = self.policy.trainable_variables
#     gradients = tape.gradient(loss, variables)
#
#     self.optimizer.apply_gradients(zip(gradients, variables))
