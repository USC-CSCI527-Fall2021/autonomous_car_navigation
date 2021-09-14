# autonomous_driving_CSCI_557

## Installation Steps
1. **Setup Python environment.** 

This will create a new python 3.6 environment named "c557".

```
make python_env
```

2. **Activate Python Environmrnt**
```
conda activate c557
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

## WEEK 4
1. DDQN with no PER
2. Default throttle and discret steer values
3. Epsilon Greedy Strategy
4. No obstactes or other actors.

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