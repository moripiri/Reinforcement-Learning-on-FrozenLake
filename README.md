# Reinforcement-Learning-on-FrozenLake
**Reinforcement Learning on FrozenLake** is a collection of jupyter files that you can learn and try basic rl algorithms in sutton's RL book.

This repo is written for people who want to quickly learn basic concepts of Reinforcement Learning with code.

**2024-11-24: Made a demo version of this repo! [link](https://rlonfrozenlake.streamlit.app/)**


---
## 💡 Features
- **Easy explanation of RL concepts**
  - This book contains key part of the book **"Reinforcement Learning: An Introduction"**([pdf](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)) by Richard S. Sutton and Andrew G. Barto.
  - The lecture **"Introduction to Reinforcement Learning with David Silver"**([link](https://www.deepmind.com/learning-resources/introduction-to-reinforcement-learning-with-david-silver)) is also referred. 
- **Interactive RL algorithm**
  - You can also run RL algorithms in **FrozenLake-v1**, a OpenAI Gymnasium environment, with hyperparmeter customization.
  - Wrapper for the environment can render not only the environment, but also **state-action value** or **model** of the algorithms.
  <p align="center">
    <img src="./Images/dynaq.gif" width=500 height=500>
  </p>
---
## ☝️ Requirements
- python >= 3.6
- gymnasium >= 0.26.1
- pygame >= 2.3.1
- tensorflow >= 2.8 (for Chapter 7)
- tensorflow-probability >= 0.22.1 (for Chapter 7)
- ipykernel
- ipython
- numpy
- matplotlib

You can install the requirements by using **Poetry**.
```bash
git clone https://github.com/moripiri/Reinforcement-Learning-on-FrozenLake.git
cd Reinforcement-Learning-on-FrozenLake

poetry install #--with ch7 #(If you want to run chapter7.ipynb, optional tensorflow dependency have to be installed.
poetry run python -m ipykernel install --user --name [virtualEnv] --display-name "[displayKernelName]"
```
where `[virtualEnv]` is the name of the python environment(ex. rl-introduction-py3.9) and `"[displayKernelName]"` is the jupyter kernel name you want (ex. frozenlake).

Then run 
```bash
poetry run jupyter notebook
```
to run jupyter files.

## ☝️ Running in Google Colab

If you run any jupyter file in **google colab**, run the following commands first.

```bash
!git clone https://github.com/moripiri/Reinforcement-Learning-on-FrozenLake.git
%cd Reinforcement-Learning-on-FrozenLake/
!pip install gymnasium[classic_control]==0.26.3
```

---
## 📖 Contents
#### Chapter1: Introduction to Reinforcement Learning 
#### Chapter2: Markov Decision Processes
#### Chapter3: Dynamic Programming
- **Policy iteration**
- **Value iteration**

#### Chapter4: Model-Free Prediction  
- **Monte-Carlo Prediction**
- **TD(0)**
#### Chapter5: Model-Free Control
- **On-policy Monte-Carlo Control**
- **SARSA**
- **Q-learning**
#### Chapter6: Eligibility Traces
- **SARSA(λ)**

#### Chapter7: Policy Gradient Methods
- **REINFORCE**
- **Actor-Critic**

#### Chapter8: Integrating Learning and Planning
- **Dyna-Q**

---
