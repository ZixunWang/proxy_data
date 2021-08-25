# prebuild proxy data
build:
	spring.submit arun --gpu --job-name=proxy_data_build "python3 build_proxy_data.py"

# test the correlation between proxy data and total data
export SAMPLER=random
export RESUME=./result/cifar10_random_avg_val_dict.pth
test:
	spring.submit arun --gpu --gres=gpu:4 --job-name=proxy_data "python3 test.py -b 201 -d cifar 10 -t 0.2 -s $(SAMPLER) --resume $(RESUME)"
