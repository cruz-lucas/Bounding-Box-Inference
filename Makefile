CUDA := 12.6.0
TAG := devel
OS := ubuntu20.04
IMAGE_NAME := bbi
CONTAINER_NAME := bbi
WORKDIR_PATH := /workspace

docker-up:
	docker-compose -f docker/docker-compose.yml up -d --build --remove-orphans

docker-down:
	docker-compose -f docker/docker-compose.yml down

q-learning:
	python train.py --config_file="goright_qlearning" --n_seeds=50 --start_seed=0

perfect:
	python train.py --config_file="goright_perfect" --n_seeds=50 --start_seed=0

expect-2:
	python train.py --config_file="goright_expected_h2" --n_seeds=1 --start_seed=0

expect-5:
	python train.py --config_file="goright_expected_h5" --n_seeds=1 --start_seed=99

sampling-2:
	python train.py --config_file="goright_sampling_h2" --n_seeds=40 --start_seed=10

sampling-5:
	python train.py --config_file="goright_sampling_h5" --n_seeds=10 --start_seed=0

bounding-box:
	python train.py --config_file="goright_bbi" --n_seeds=50 --start_seed=0

linear-bbi:
	python train.py --config_file="goright_bbi_linear" --n_seeds=50 --start_seed=0

tree-bbi:
	python train.py --config_file="goright_bbi_tree" --n_seeds=50 --start_seed=0

neural-bbi:
	python train.py --config_file="goright_bbi_neural" --n_seeds=50 --start_seed=0

# Clean up by removing the Docker image
clean:
	docker rmi $(IMAGE_NAME)
