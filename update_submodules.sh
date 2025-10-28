#!/usr/bin/env bash
set -euo pipefail

# 简介：
# 将本仓库中所有子模块更新到 .gitmodules 配置的分支最新提交。
# - 默认并发 4 个任务；可通过环境变量 JOBS 覆盖（例如 JOBS=8）。
# - 默认会取消浅拷贝并抓取全部分支/标签（可通过 FULL=0 关闭）。
# - 默认会将子模块切换到配置分支上（避免 detached HEAD，可通过 ATTACH=0 关闭）。
# - 可选：设置 COMMIT=1 自动提交子模块指针更新。
#
# 用法：
#   chmod +x ./update_submodules.sh
#   ./update_submodules.sh                # 仅更新
#   JOBS=8 ./update_submodules.sh         # 指定并发
#   COMMIT=1 ./update_submodules.sh       # 更新后自动提交指针
#   FULL=0 ./update_submodules.sh         # 关闭“取消浅拷贝 + 抓取全部分支/标签”

JOBS="${JOBS:-4}"
COMMIT="${COMMIT:-0}"
FULL="${FULL:-1}"
ATTACH="${ATTACH:-1}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "错误：当前目录不是 Git 仓库" >&2
  exit 1
fi

echo "[1/3] 同步子模块配置 (.gitmodules -> .git/config) ..."
git submodule sync --recursive

echo "[2/3] 更新到配置分支的最新提交（递归） ..."
# 关键：--remote 会按 .gitmodules 的 branch= 拉取对应远端分支的最新提交
# 不加 --depth，避免浅克隆；如需浅克隆可自行加 --depth=1
git submodule update --init --remote --recursive --jobs="${JOBS}"

echo "[2b] 后处理（去浅化 + 附着到分支） ..."
# 合并处理：根据 FULL/ATTACH 两个开关完成去浅化与附着到分支，减少多余 fetch
env FULL="${FULL}" ATTACH="${ATTACH}" git submodule foreach --recursive '
  set -e
  # 解析目标分支（优先 .gitmodules，其次 origin/HEAD，最后回退 main）
  BR=$(git config -f "$toplevel/.gitmodules" "submodule.$name.branch" || true)
  if [ -z "$BR" ]; then
    DEF=$(git symbolic-ref -q --short refs/remotes/origin/HEAD 2>/dev/null || true)
    if [ -n "$DEF" ]; then BR=${DEF#origin/}; else BR=main; fi
  fi

  if [ "$FULL" = "1" ]; then
    # 抓取所有分支/标签，并尽量补全历史
    git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
    git fetch origin --tags --prune --progress "+refs/heads/*:refs/remotes/origin/*" || true
    if [ "$(git rev-parse --is-shallow-repository 2>/dev/null || echo false)" = "true" ]; then
      git fetch --unshallow --tags --prune --progress || \
      git fetch origin --deepen=2147483647 --tags --prune --progress || true
    fi
  fi

  if [ "$ATTACH" = "1" ]; then
    if git show-ref --verify --quiet "refs/remotes/origin/$BR"; then
      git checkout -B "$BR" "origin/$BR" --quiet || git checkout "$BR" --quiet || true
      git branch --set-upstream-to="origin/$BR" "$BR" >/dev/null 2>&1 || true
      echo " - $name attached: $(git rev-parse --abbrev-ref HEAD || echo detached)"
    else
      echo " ! $name: missing origin/$BR"
    fi
  fi

  if [ "$FULL" = "1" ]; then
    if [ -f .git/shallow ] || [ "$(git rev-parse --is-shallow-repository 2>/dev/null || echo false)" = "true" ]; then
      echo " ! $name still shallow"
    else
      echo " - $name full history"
    fi
  fi
'

echo "[3/3] 当前子模块指针："
git submodule status || true

echo "完成：子模块已更新到配置分支的最新提交。"

if [[ "${COMMIT}" == "1" ]]; then
  echo "[commit] 提交父仓库中的子模块指针 ..."
  # 防御：若存在未合并文件，则中止提交，避免报错体验
  if git diff --name-only --diff-filter=U | grep -q .; then
    echo "存在未解决的冲突，已跳过自动提交。请先解决冲突后手动提交。" >&2
    exit 0
  fi
  # 列出所有子模块路径，添加到提交
  if paths=$(git submodule--helper list 2>/dev/null | awk '{print $4}'); then
    if [[ -n "${paths}" ]]; then
      git add ${paths}
      git commit -m "chore: bump submodules to configured branches"
      echo "已提交子模块指针更新。"
    else
      echo "没有检测到子模块路径，跳过提交。"
    fi
  else
    echo "警告：git submodule--helper 不可用，跳过自动提交。"
  fi
fi
