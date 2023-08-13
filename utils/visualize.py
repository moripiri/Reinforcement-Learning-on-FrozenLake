import matplotlib.pyplot as plt
import numpy as np

def visualize_model(model: np.ndarray) -> None:
    pass

def visualize_q(q_func: np.ndarray) -> None:
    pass

def visualize_v(v_func: np.ndarray) -> None:
    pass

def visualize_policy(policy: np.ndarray) -> None:
    pass


def plot_values(V):
    # reshape value function
    V_sq = np.reshape(V, (4,4))

    # plot the state-value function
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(V_sq, cmap='cool')
    for (j,i),label in np.ndenumerate(V_sq):
        ax.text(i, j, np.round(label, 5), ha='center', va='center', fontsize=14)
    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.title('State-Value Function')
    plt.show()