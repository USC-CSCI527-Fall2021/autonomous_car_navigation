python_env:
	conda create -n c527 python=3.6 -y

carla_env:
	pip install -r gym-carla/requirements.txt && \
	pip install -e gym-carla && \
	wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz && \
	mkdir CARLA_0.9.6 && tar -xf CARLA_0.9.6.tar.gz -C CARLA_0.9.6 && rm CARLA_0.9.6.tar.gz

run_display:
	./CARLA_0.9.6/CarlaUE4.sh -windowed -carla-port=2000 &

run_non_display:
	DISPLAY= ./CARLA_0.9.6/CarlaUE4.sh -opengl -carla-port=2000 &

test:
	python gym-carla/test.py
	
