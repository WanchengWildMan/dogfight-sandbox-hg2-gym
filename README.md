# dogfight-sandbox-hg2-gym
Reinforcement environment of an air to air combat sandbox, created in Python 3 using the HARFANG 3D 2 framework. Integrated PPO as an example.
# Structure 
```env.py``` The environment for algorithm to interact with the game using interface ```step```.
# Configurations
```agent_type_action_mapping.json``` Actions of each type of agents. Remember to correspond the ```num_agents``` and ```num_agent_types```.
```state_config.json``` Methods for obtaining states from ```Machines```
