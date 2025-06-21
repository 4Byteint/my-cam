#!/bin/bash

# ✅ 設定參數
CAN_IF="can0"
BITRATE=1000000

echo "嘗試關閉 $CAN_IF（若已存在）..."
sudo ip link set $CAN_IF down 2>/dev/null

echo "設定 $CAN_IF 為 SocketCAN 介面，bitrate = $BITRATE"
sudo ip link set $CAN_IF type can bitrate $BITRATE

echo "啟動 $CAN_IF..."
sudo ip link set $CAN_IF up

echo "CAN 介面啟用完成！目前狀態："
ip -details link show $CAN_IF | grep -A5 "$CAN_IF"

# ✅ 可加上測試指令（如有資料傳送裝置）
# echo "📡 等待 CAN 資料..."
# candump $CAN_IF
