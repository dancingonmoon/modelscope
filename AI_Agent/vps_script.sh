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

# 以下为安装V2Ray 梯子服务器搭建：
# VPS服务器需要进行服务器时间同步校对时就需要用到时间同步组件ntp，使服务器时间保持跟全球网络同步
apt-get install ntp ntpdate -y
#安装完成后，先停止当前 VPS 服务器的 NTP 服务
service ntpd stop
#然后再使当前 VPS 服务器的时间与时间服务器进行同步
ntpdate us.pool.ntp.org
#最后启动 NTP 服务
service ntpd start

#执行V2Ray一键安装脚本
bash <(wget -qO- -o- https://git.io/v2ray.sh)