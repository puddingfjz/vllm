# code structure
benchmarks
    my_bench_multimodel_throughput.py:      end-to-end multimodels
    benchmark_throughput.py:                end-to-end single model [貌似这个函数目前还不能通过命令行控制backend是vllm还是ours，所以还是要在一开始通过环境变量修改] [删掉了一开始设置cuda visible环境变量的部分，现在必须要在命令行前面输入关于cuda visible的环境变量才可以]

    output_length_sampler.py:               sampling output length
    fake_scheduling.py:                     doing fake scheduling to estimate cost
    
    <!-- 先重点看这两部分 -->
    construct_cost_model.py:                run this file to collect data to build cost model
    my_per_iter_latency_estimator.py:       ? calling the cost model ? 看看这个文件
                                            1. 需要额外补充prepare_model_weights的时间，其实就是initialization的开销，但是
                                            这个开销其实感觉还是收到RAM的影响的，如果之前model已经被load过一次的话，这部分的开销才会
                                            比较稳定。还是要修改一下这部分的cost的measure的函数，最好能自动获得.
                                            -->其实测的就是init LLM instance 的时间。已经补充上这部分的script了。
    
    model_coeff_database.py:                stores the coefficients for each model to compute the flops
                                            obtained by running comp_model_size.py

    comp_model_size.py:                     run this file (1) to get the model parameter sizes and store them in
                                            model_size_database.py
                                            and (2) to get the model coefficients to compute flops and store them in model_coeff_database.py

    model_size_database.py:                 stores the model parameter sizes
                                            obtained by running comp_model_size.py

    model_initcost_database.py              stores the model init costs
                                            obtained by running comp_model_size.py
                                            NOTE: every time introducing a new model (i.e., the models except those used in LLM-Blender), run comp_model_size.py 
                                                for that model.

    search_exec_plans.py:                   generate multi-model schedule plan
                                            增加data parallel的支持的时候所有改动的代码都标注了data parallel
                                            为了支持data parallel：
                                                所有exec_plan都保留所有dp worker的inference metadata

                                            NOTE: 
                                                目前我们其实并不打算支持cache gpu了；
                                                关于cache gpu，我们暂时假定不同的tp worker 不一定使用相同的cache gpu，
                                                不同的dp worker也不一定使用相同的cache gpu。
                                                ！！！！但是关于这部分Memory use的implementation其实是假定所有tp worker都使用相同的cache gpu的，这块还需要改进！！！！！


    schedule_multi_model.py:                schedule the end-to-end inference
                                            parameters: 
                                                (1) gen_execplans_baseline: "naive" -> use all gpu cards to run a model
                                                (2) search_method_baseline: "naive" -> using the greedy baseline: _get_best_model_schedule_greedy_baseline_adapted_from_MuxServe

    my_llm_infer_worker.py                  把每个data parallel ray actor的inference代码放到这个文件来了。
                                            bench_throughput.py run_vllm函数会调用这个文件的函数。
    

    multimodel_scheduler.py                 使用ray actor来支持data parallelism的话，会通过另一个额外的ray actor来 
                                            message passer，然后主data parallel actor通过这个passer来通知别的data parallel actor来中断inference。
                                            等主data parallel actor收到了所有data parallel actor inference的结果之后，再mark reschedule过程的结束。





在服务器上启动实验环境的代码-------------------------
在lccpu27上做vLLM的实验：
setenv HOME "/ssdexport/data/jingzhi/cutlass"
# 还是要重新安装anaconda，不能使用miniconda
# setenv PATH "/ssdexport/data/jingzhi/cutlass/miniconda3/bin:$PATH"
setenv PATH "/ssdexport/data/jingzhi/cutlass/anaconda3/bin:$PATH"
conda create -n vllm python=3.9 -y

bash
# conda activate vllm
conda activate new_vllm
conda activate vllm_baseline
export PS1='($CONDA_DEFAULT_ENV):\u@\h:\w>'
# export PATH=/usr/local/cuda-11.7.0/bin:$PATH # 升级了vllm之后需要换到cuda12
export PATH=/usr/local/cuda-12.3.1/bin:$PATH
# 更换g++ 版本否则编译不通过
export PATH=/usr/local/GNU/gcc-11.2.0/bin:$PATH
# 需要更新 pyarrow
pip3 install -U pyarrow 
pip3 install -e . 

在27上启动ray会有资源不足的问题，需要再运行程序之前运行以下命令
ulimit -u 65536 

#------

在lccpu28上做vLLM的实验：
screen -h 5000000 -S vllm
setenv HOME "/ssddata/jingzhi"
# setenv PATH "/ssddata/jingzhi/miniconda/miniconda3:$PATH"
# setenv PATH "/ssddata/jingzhi/miniconda/miniconda3/pkgs/conda-23.5.2-py39h06a4308_0/bin:$PATH" # 这样设置不行，直接无法运行conda命令
# setenv PATH "/ssddata/jingzhi/miniconda/miniconda3/pkgs/conda-23.5.2-py39h06a4308_0/condabin:$PATH" 

# setenv CONDARC "/ssddata/jingzhi/.condarc"
# conda.exe config --file $CONDARC --add envs_dirs /ssddata/jingzhi/miniconda/miniconda3/envs
# conda.exe config --file $CONDARC --add pkgs_dirs /ssddata/jingzhi/miniconda/miniconda3/cache_pkgs
# conda.exe create --prefix /ssddata/jingzhi/miniconda/miniconda3/envs/vllm python=3.9 -y

#决定直接安装anaconda而不是miniconda
setenv PATH "/ssddata/jingzhi/anaconda3/bin:$PATH"
conda create -n vllm python=3.9 -y

# 要进入bash环境之后才能切换conda环境，不知道为啥
bash
export PS1='($CONDA_DEFAULT_ENV):\u@\h:\w>'
conda activate vllm
conda deactivate

export PATH=/usr/local/cuda-11.7.0/bin:$PATH
# 此时的path：
/usr/local/cuda-11.7.0/bin:/ssddata/jingzhi/anaconda3/envs/vllm/bin:/ssddata/jingzhi/anaconda3/condabin:/ssddata/jingzhi/anaconda3/bin:/homes/jfangak/bin:/homes/jfangak/bin/x86_64:/sbin:/usr/sbin:/bin:/usr/bin:/usr/local/bin:/usr/local/share/bin:.

# 需要更新 pyarrow
pip3 install -U pyarrow 

pip3 install -e . 

pip3 install --target=/ssddata/jingzhi/vLLM/my_python_libs2 -e . 


# 如果需要重新编译
pip3 install -e . 



# 貌似重新进入putty的时候只需要执行以下命令 （为啥进入bash环境之后anaconda会自动生效？）
screen -h 5000000 -S vllm
setenv HOME "/ssddata/jingzhi"
bash
export PS1='($CONDA_DEFAULT_ENV):\u@\h:\w>'
# conda activate vllm
conda activate new_vllm
# export PATH=/usr/local/cuda-11.7.0/bin:$PATH (升级vllm之后得用cuda12)
export PATH=/usr/local/cuda-12.1.1/bin:$PATH



# 需要运行Gurobi的时候应该设置环境变量(但是貌似不设也行)
export GRB_LICENSE_FILE='/ssdexport/data/jingzhi/cutlass/gurobi.lic'


# 测试bandwidth
./bandwidthTest –device=2 --memory=pinned --mode=shmoo --htod


# 查看 nvlink 的信息
nvidia-smi nvlink -h

# 为了使用vscode设置软连接
cd /homes/jfangak
rm -rf .vscode-server
ln -s /ssddata/jingzhi/.vscode-server_my/ .vscode-server




# 之前把huggingface缓存的位置设成了 
export HF_DATASETS_CACHE=/export/data/jingzhi/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/export/data/jingzhi/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/export/data/jingzhi/.cache/huggingface/hub

# 如果需要获得huggingface的access token
#pip install "huggingface_hub[cli]"
输入命令：huggingface-hub login  或者 huggingface-cli login
然后输入token







# ##############################################################################################
# ##############################################################################################
# ##############################################################################################
# ##############################################################################################
# 在zxcpu3 上做vllm的实验
0. 下载cuda并安装
export HOME=/ssddata/jingzhi
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
sh cuda_12.4.1_550.54.15_linux.run

1. 安装完cuda之后配置环境变量
# export CUDA_HOME=/ssddata/jingzhi/cuda-12.4/
# export LIBRARY_PATH=/ssddata/jingzhi/cuda-12.4/lib64:$LIBRARY_PATH

export PATH=/ssddata/jingzhi/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/ssddata/jingzhi/cuda-12.4/lib64:$LD_LIBRARY_PATH

<!-- 安装cuda的summary --------------------------------------------------------------->
===========
= Summary =
===========

Driver:   Not Selected
Toolkit:  Installed in /ssddata/jingzhi/cuda-12.4/

Please make sure that
 -   PATH includes /ssddata/jingzhi/cuda-12.4/bin
 -   LD_LIBRARY_PATH includes /ssddata/jingzhi/cuda-12.4/lib64, or, add /ssddata/jingzhi/cuda-12.4/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /ssddata/jingzhi/cuda-12.4/bin
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 550.00 is required for CUDA 12.4 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

<!-- 安装cuda的summary END ------------------------------------------------------------>

2. 安装anaconda
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
shasum -a 256 ./Anaconda3-2024.06-1-Linux-x86_64.sh
bash ./Anaconda3-2024.06-1-Linux-x86_64.sh
export PATH=/ssddata/jingzhi/anaconda3/bin:$PATH


3. 配置new_vllm环境
conda create -n new_vllm python=3.9 -y
source /ssddata/jingzhi/anaconda3/bin/activate
conda init
conda activate new_vllm
conda deactivate


4. 下载并编译vllm
# 需要更新 pyarrow
pip3 install -U pyarrow 
# 下载 einops
pip install einops

# 编译vllm
git clone https://github.com/puddingfjz/vllm.git
cd vllm
pip3 install -e . 

安装最新版的时候或许需要
CUDACXX=/ssddata/jingzhi/cuda-12.4/bin/nvcc pip install -e  .


# 指定numpy, torch版本
pip install numpy==1.26.3

# pip tmp空间不足的时候
export TMPDIR=$HOME/tmp


# create some folds
cd benchmarks
mkdir my_dummy_requests
mkdir test_search
mkdir test_end2end_schedule
mkdir Cost_Model_per_iter_zxcpu

5. =====================================================
安装完之后重新进入环境的步骤
screen -h 5000000 -S vllm
export HOME=/ssddata/jingzhi
export PATH=/ssddata/jingzhi/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/ssddata/jingzhi/cuda-12.4/lib64:$LD_LIBRARY_PATH
# export PATH=/ssddata/jingzhi/anaconda3/bin:$PATH [这个不需要了，因为已经conda init加到环境变量里了]
bash
conda activate new_vllm

