# [RE] S2V-DQN

## Acknowledge

Code Source: https://github.com/Hanjun-Dai/graph_comb_opt  
Paper link: https://arxiv.org/abs/1704.01665

## 1. Installation

**Well tested under Ubuntu 18.04.6 LTS, macOS Monterey 12.0.1**

We **strongly** recommend that using Docker to build an image with the Dockerfile provided. Dockerfile contains all the
required installations except NVDIA toolkit.

* Download NVIDIA Docker toolkit

```sh
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit nvidia-container-runtime
sudo systemctl restart docker
```

* Build an Image

```sh
docker build -t "s2v-dqn:test" .
```

Addition to building an image with the Dockerfile by yourself, we also provide a [Docker image](https://hub.docker.com/r/710949644/s2v-dqn) at Docker Hub,
and you can pull the image and deploy the project directly.

```sh
docker pull 710949644/s2v-dqn
```

* Start

```sh
docker run -ti --gpus all -v $ABSOLUTE_PATH_OF_THE_PROJECT:/root/S2V-DQN -v $ABSOLUTE_PATH_OF_DATA_DIRECTORY:/root/data s2v-dqn:test bash
```

**If you don't use docker**:  
Please refer to source code to get more details about installing the required packages.

## 2. Build Libraries

We need to build graphnn libary and mvc_lib library.

1. build graphnn  
   navigate to graphnn,
    ```sh
    # if use GPU
    make -f Makefile
    # if use CPU
    make -f Makefile_cpu
    ```
   Check the path of INTEL package in case required packages can not be loaded correctlly.

2. build mvc lib  
   navigate to code/realworld_s2v_mvc/
    ```sh
    # select a makefile to make according your platform
   make 
    ```


## 3. Training
Execute the command to train with default parameter setting:
```sh
cd code/realworld_s2v_mvc
bash run_nstep_dqn.sh $dataset
```
We provide trained models that trained on dataset memetracker or BrightKite.  
Download from [HERE](https://drive.google.com/file/d/1uro-k4xRrWFShnCpxhcDUWlGCsaqFkkv/view?usp=sharing), and then put the files under code/realworld_s2v_mvc/results/. 

## 4. Test
Execute the command to test the model with default parameter setting:
```sh
cd code/realworld_s2v_mvc
bash run_eval.sh $dataset
```
The results file would be generated to code/realworld_s2v_mvc/results/testphase.

# 5. Output Format
The results file would be generated at a directory realworld_s2v_mvc/results/testphase/$dataset/.
The format of output is
```markdown
$coverage $runtime
$solution_set
```

