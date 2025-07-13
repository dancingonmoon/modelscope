#!/bin/bash

# "为了让这个文件能够作为脚本执行，你需要给它赋予执行权限: chmod +x V2RayServer.sh"
# "现在你可以通过在终端中输入脚本的路径来执行它：./V2RayServer.sh"

echo "VPS服务器需要进行服务器时间同步校对时就需要用到时间同步组件ntp，使服务器时间保持跟全球网络同步"
apt-get install ntp ntpdate -y
echo "安装完成后，先停止当前 VPS 服务器的 NTP 服务"
service ntpd stop
echo "然后再使当前 VPS 服务器的时间与时间服务器进行同步"
ntpdate us.pool.ntp.org
echo "最后启动 NTP 服务"
service ntpd start

echo "执行V2Ray一键安装脚本"
bash <(wget -qO- -o- https://git.io/v2ray.sh)

echo "现在可以尝试一下输入 v2ray 回车，即可管理 V2Ray"
echo "将v2ray 配置的端口暴露给外网： sudo ufw allow PortNo"
echo "查看某个IP地址下的指定端口，全球访问情况：https://tcp.ping.pe/45.76.163.104:80"

# V2RayServer.sh 文件夹位置: "E:/Python_WorkSpace/modelscope/AI_Agent\V2RayServer.sh"
# scp "E:/Python_WorkSpace/modelscope/AI_Agent\V2RayServer.sh" root@ :/root

