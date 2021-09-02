# autonomous_driving_CSCI_557

## Installation Steps
1. **Setup Python environment.** 

This will create a new python 3.6 environment named "c557".

```
$ make python_env
```

2. **Activate Python Environmrnt**
```
$ conda activate c557
```

3. **Setup Carla Environment**

This will install all required library, download CARLA 0.9.6, extract it
```
$ export PYTHONPATH=$PYTHONPATH:$PWD/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg
```
	
## Usage
1. **Set paths to use carla environment**
```
$ make set_path
```

2. **Start Carla Environment**

In Display mode, starts a window
```
$ make run_display
``` 

In Non Display mode, must be accessed via python api.
```
$ make run_non_display

```

3. **Test Carla Environment**
```
$ make test
```

## Contributors

1. [Aditya Jain](https://adityajain.me)
2. Harshita Bhorshetti
3. Isha Chaudhari
4. Devanshi Desai
5. Saichand Duggirala
6. Adwaita Jadhav
7. Aditi Jain

## References

1. [CARLA Simulator](https://github.com/carla-simulator/carla)
2. [Gym Wrapper for CARLA](https://github.com/cjy1992/gym-carla)