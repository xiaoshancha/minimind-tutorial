#!/usr/bin/env bash
# 本地一键部署：构建 -> 通过 SSH 把 dist 上传到阿里云子路径
# 依赖：已把 ~/.ssh/id_ed25519_penguin 的公钥添加到阿里云 ecs 的 admin 用户
# 用法：bash deploy-local.sh
set -euo pipefail

HOST="123.56.2.125"
USER="admin"
PORT="22"
KEY="$HOME/.ssh/id_ed25519_penguin"
REMOTE_ROOT="/www/wwwroot/tutorial.baimuyuan.online"
SUBDIR="minimind-tutorial"

echo "[1/4] 构建静态站 ..."
npm run build

echo "[2/4] 校验构建产物 ..."
npm run verify

echo "[3/4] 清理服务器旧文件并创建目录 ..."
ssh -i "$KEY" -p "$PORT" "$USER@$HOST" "mkdir -p '$REMOTE_ROOT/$SUBDIR' && rm -rf '$REMOTE_ROOT/$SUBDIR'/*"

echo "[4/4] 上传 dist 到 $HOST:$REMOTE_ROOT/$SUBDIR ..."
scp -i "$KEY" -P "$PORT" -r .vitepress/dist/* "$USER@$HOST:$REMOTE_ROOT/$SUBDIR/"

echo "完成！访问 https://tutorial.baimuyuan.online/$SUBDIR/"
