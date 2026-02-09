#!/bin/bash
# 监控训练进度
LOG_DIR="logs/stage1_hit_only"

echo "========== 训练监控 =========="
echo "日志目录: $LOG_DIR"
echo ""

# 持续监控最新的日志
if [ -f "$LOG_DIR/progress.csv" ]; then
    echo "📊 最新训练进度："
    tail -20 "$LOG_DIR/progress.csv"
else
    echo "⚠️  progress.csv 还未生成，等待训练开始..."
fi

echo ""
echo "💡 实时监控命令："
echo "  tail -f $LOG_DIR/progress.csv"
echo ""
echo "📈 TensorBoard 命令："
echo "  tensorboard --logdir $LOG_DIR/tensorboard --port 6006"
