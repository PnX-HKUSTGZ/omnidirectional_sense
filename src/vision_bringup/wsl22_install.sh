#!/bin/bash

curl http://baidu.com
# check internet connection
#ping -c1 -w2 baidu.com > /dev/null
if [[ $? != 0 ]];then     
    echo "No internet connection. Try to connect to the internet first then run this script again."
    exit 0
else
    echo "Internet connected."
fi

echo -e "\e[32mATTENTION:\e[0m"
echo "during the installation, you may need to input your password for sudo command"
echo "you also need to input 'y' or 'n' for some options"
read -p "NOW, press [q/Q] to quit, [enter] or [y/Y] to proceed: " input
if [[ "$input" =~ ^[qQ]$ ]]; then
    echo "Exiting..."
    exit 0
elif [[ "$input" =~ ^[yY]$ || -z "$input" ]]; then
    echo "Proceeding..."
else
    echo "Invalid input. Exiting..."
    exit 1
fi

# change apt source
sudo mv /etc/apt/sources.list /etc/apt/sources.list.bake
sudo sh -c "echo 'deb https://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse

# deb https://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse
# deb-src https://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse

deb https://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse
# deb-src https://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse

deb https://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse
#deb-src https://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse

deb https://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse
# deb-src https://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-proposed main restricted universe multiverse' > /etc/apt/sources.list"
# deb https://mirrors.aliyun.com/ubuntu/ jammy-proposed main restricted universe multiverse
# deb-src https://mirrors.aliyun.com/ubuntu/ jammy-proposed main restricted universe multiverse

echo -e "\e[32minstall basic tools...\e[0m"
sudo apt update && sudo apt upgrade -
sudo apt install gedit libopencv-dev libopencv-contrib-dev libceres-dev libeigen3-dev build-essential gdb git cmake htop curl gnupg2 wget -y

# install ros2
echo -e "\e[32minstall ROS2...\e[0m"
sudo wget -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] https://mirrors.aliyun.com/ros2/ubuntu jammy main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update && sudo apt install ros-humble-desktop ros-dev-tools -y
sudo apt install ros-humble-asio-cmake-module
dpkg -L ros-humble-asio-cmake-module
if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
fi

# 手动模拟 rosdep init
sudo mkdir -p /etc/ros/rosdep/sources.list.d/
sudo curl -o /etc/ros/rosdep/sources.list.d/20-default.list https://mirrors.aliyun.com/github-raw/ros/rosdistro/master/rosdep/sources.list.d/20-default.list
# 为 rosdep update 换源
export ROSDISTRO_INDEX_URL=https://mirrors.aliyun.com/rosdistro/index-v4.yaml
sudo apt install python3-rosdep2 -y
rosdep update

# 每次 rosdep update 之前，均需要增加该环境变量. 为了持久化该设定，将其写入 .bashrc 中
if ! grep -q "export ROSDISTRO_INDEX_URL=https://mirrors.aliyun.com/rosdistro/index-v4.yaml" ~/.bashrc; then
    echo 'export ROSDISTRO_INDEX_URL=https://mirrors.aliyun.com/rosdistro/index-v4.yaml' >> ~/.bashrc
fi

# add some convinient alias,prevent duplicate alias
if ! grep -q "alias mdc='mkdir build && cd build'" ~/.bashrc; then
    echo "alias cb='colcon build --symlink-install'
    alias mdc='mkdir build && cd build'
    alias mdb='mkdir build && cd build && cmake .. && make -j'
    alias c.='cd ..'
    alias c..='cd ../..'
    alias rd='rm -r'
    alias md='mkdir -pv'
    alias h=history
    alias g1='git clone --depth 1'
    " >> ~/.bashrc
fi


# remove ln from sh->dash, use sh->bash
sudo rm /bin/sh
sudo ln -s /bin/bash /bin/sh

#install .deb files
#wget https://github.com/coder/code-server/releases/download/v4.22.1/code-server_4.22.1_amd64.deb 
# wget https://packages.microsoft.com/repos/vscode/pool/main/c/code/code_1.94.0-1727878498_amd64.deb 
# wget https://packages.microsoft.com/repos/vscode/pool/main/c/code/code_1.87.2-1709912201_amd64.deb 
# wget https://packages.microsoft.com/repos/vscode/pool/main/c/code/code_1.94.1-1728111316_amd64.deb          

# #install network-manager
# sudo apt update
# sudo apt install network-manager

# install dependencies from .deb files
echo -e "\e[33mNow we turn to deb installation. Make sure these .deb(s) are in current dir \e[0m"
read -p "press [Y/y] to install or any other key to skip : " input
if [[ ! "$input" =~ ^[yY]$ ]]; then
    echo "Skip deb installation..."
else
    read -p $'\e[32m install Hik MVS tools[M/m], just runtime pack[R/r], do not install any[N/n]: \e[0m' install
    if [ "$install" == "M" ] || [ "$install" == "m" ]; then
        sudo dpkg -i MVS-2.1.2_x86_64_20231225.deb 
    elif [ "$install" == "R" ] || [ "$install" == "r" ]; then
        sudo dpkg -i MvCamCtrlSDK_Runtime-4.3.0_x86_64_20231225 -y
    else
        echo "Skipping Hik MVS tools installation."
    fi

    read -p $'\e[32m install NoMachine remote desktop? (y/n): \e[0m' install
    if [ "$install" == "y" ]; then
        sudo dpkg -i nomachine_8.11.3_4_amd64.deb
    else
        echo "Skipping NoMachine installation."
    fi

    read -p $'\e[32m install VSCode & code-server? (y/n): \e[0m' install
    if [ "$install" == "y" ]; then
        sudo dpkg -i code-server_4.22.1_amd64.deb
        sudo dpkg -i code_1.87.2-1709912201_amd64.deb
        sudo dpkg -i code_1.94.0-1727878498_amd64.deb
        sudo apt-get install -f
    else
        echo "Skipping VSCode installation."
    fi
fi

# pull and build rm_vision
read -p $'\e[32mPull the latest code and build? (y/n): \e[0m' pull_code
if [ "$pull_code" == "y" ]; then
    cd ~
    source ~/.bashrc

    if [ -d "pnx_ws" ]; then
        echo -e "\e[34m'pnx_ws' directory already exists. try pull newest commits.\e[0m"
        cd pnx_ws/src
        for d in ./*; do
            if [ -d "$d" ]; then
                cd "$d"
                if [ -d ".git" ]; then
                    echo "Updating $d"
                    git pull
                fi
                cd ..
            fi
        done
    else
        echo "cloning the latest code..."
        mkdir -p pnx_ws/src && cd pnx_ws/src
        git clone https://github.com/PnX-HKUSTGZ/auto-aim.git --depth 1
        git clone https://github.com/PnX-HKUSTGZ/vision_bringup.git --depth 1
        git clone --depth 1  https://github.com/PnX-HKUSTGZ/rm_serial_driver.git
        git clone --depth 1  https://github.com/PnX-HKUSTGZ/rm_gimbal_description.git
        git clone --depth 1  https://github.com/PnX-HKUSTGZ/hik-driver.git
    fi

    cd ..
    echo "building..."
    rosdep install --from-paths src --ignore-src -r -y
    colcon build --symlink-install --packages-up-to rm_vision_bringup
    source ~/pnx_ws/install/setup.bash
    echo 'source ~/pnx_ws/install/setup.bash' >> ~/.bashrc
else
    echo "Skipping code pull and build."
fi

echo " "
echo "well done"
echo "To dev in the terminal, please run 'source ~/.bashrc' to enable the environment"

rm -rf build/ install/ log/
colcon build
