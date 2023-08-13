import gymnasium as gym
import matplotlib
import IPython
from IPython import display
import matplotlib.pyplot as plt

class JupyterRender(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def render(self):
        img = plt.imshow(super.render())  # prepare to render the environment by using matplotlib and ipython display
        display.display(plt.gcf())
        display.clear_output(wait=True)
          
        
        
    



if __name__ == '__main__':
    env = gym.make("FrozenLake-v1", render_mode='rgb_array', is_slippery=False)  # define the environment.
    env = JupyterRender(env)
