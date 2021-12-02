# autonomous_driving_CSCI_557

**Website**: [https://autodynamics.webflow.io](https://autodynamics.webflow.io) 

## Installation Steps
1. **Setup Python environment.** 

This will create a new python 3.6 environment named "c527".

```
make python_env
```

2. **Activate Python Environmrnt**
```
conda activate c527
```

3. **Setup Carla Environment**

This will install all required library, download CARLA 0.9.6, extract it.
```
make carla_env
```
	
## Usage
1. **Set paths to use carla environment**
```
export PYTHONPATH=$PYTHONPATH:$PWD/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg
```

2. **Start Carla Environment**

In Display mode, starts a window
```
make run_display
``` 

In Non Display mode, must be accessed via python api.
```
make run_non_display

```

3. **Test Carla Environment**
```
make test
```

## Models and Paths
1. DQN & DDQN - [Week 4 - DQN](WEEK%204%20-%20DQN)
2. VAE training - [vae/VAE_training.ipynb](vae/VAE_training.ipynb)
3. PPO with VAE - [Midterm - PPO final/ppo_vae.py](Midterm%20-%20PPO%20final/ppo_vae.py) 
4. PPO with Cropped Image View - [Midterm - PPO final/ppo_vae.py]
(Midterm%20-%20PPO%20final/ppo_no_vae)
5. Imitation Learning - [Classifier/classifier_carla_2.ipynb](Classifier/classifier_carla_2.ipynb)
6. PPO using baseline imitation learning - [PPO-immitation/ppo-imitation-only-steer.ipynb](PPO-immitation/ppo-imitation-only-steer.ipynb)


## Contributors

1. [Aditya Jain](https://adityajain.me)
2. [Harshita Bhorshetti](https://github.com/HarshitaBhorshetti30)
3. [Isha Chaudhari](https://github.com/isha31)
4. [Devanshi Desai](https://github.com/DevanshiDesai)
5. [Saichand Duggirala](https://github.com/dsaichand3)
6. [Adwaita Jadhav](https://github.com/adwaita1)
7. [Aditi Jain](https://github.com/aditi1208)

## References

1. [CARLA Simulator](https://github.com/carla-simulator/carla)
2. [Gym Wrapper for CARLA](https://github.com/cjy1992/gym-carla)