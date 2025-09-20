#!/bin/bash

# 定义主文件夹路径
MAIN_DIR="evaluation"
cd $MAIN_DIR
SOLUTION=${1:-llm}

# 定义日志文件路径
LOG_FILE="/home/yfwang/wyf/setting1/hard/run.log"

# 清空旧的日志文件
> "$LOG_FILE"

# 遍历主文件夹下的所有子文件夹
for SUB_DIR in */; do
    echo "------------------------------------------------------------------------------------------" >> "$LOG_FILE"
    cd $SUB_DIR

    timeout 600s python run.py --solution "$SOLUTION" 2>> "$LOG_FILE"
    if [ $? -eq 124 ]; then
        # timeout 命令返回 124 表示超时
        echo "Successfully completed $SUB_DIR" >> "$LOG_FILE"
    elif [ $? -ne 0 ]; then
        # 其他非零状态码表示运行失败
        echo "Error occurred for $SUB_DIR" >> "$LOG_FILE"
    else
        # 成功执行
        echo "Successfully completed $SUB_DIR" >> "$LOG_FILE"
    fi

    echo "------------------------------------------------------------------------------------------" >> "$LOG_FILE"
    cd ..
done