all:
	nvcc -std=c++17 runner.cu -o runner -lcublas
