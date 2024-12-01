#!/bin/bash
# 安装anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
# conda --version
source ~/.bashrc
conda --version
# 创建conda环境
conda create -n myenv python=3.12
conda activate myenv
# 升级conda
conda update -n base -c defaults conda

# pip install -q -U google-generativeai
# pip install -U gradio
# pip install -U zhipuai

# 允许某个端口可以被外部访问；
sudo ufw allow 7860
