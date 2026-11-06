# 数据审核工作台

一个轻量的本地审核工具：读取 I2RT NEDF3 v2 的原始 episode，生成双腕回放和技术检查，再逐条作出人工决定。不需要训练框架、设备SDK或旧仓库的Python代码。

## 开始使用

使用Python 3.10或以上，在本目录安装依赖：

```bash
python -m pip install -r requirements.txt
```

Windows双击 **start_review.cmd** 打开已有任务；双击 **prepare_task.cmd** 选择原始任务并生成回放。保持启动窗口开启，Ctrl+C停止服务。默认地址为 `http://127.0.0.1:8776/`。

启动脚本优先使用环境变量 `REVIEW_PYTHON` 指定的解释器，其次本目录`.venv`，否则当前环境的`python`。Linux可运行`bash start_review.sh`。

本机已有的“选中蓝色积木，放入蓝色杯子”任务的回放、检查结果和审核记录已复制到新目录；旧工具及原始数据未改动。

## 换一批原始数据

原始结构：

```text
数据根目录/
  任务名/
    episode_0000/
      metadata.json
      episode_stats.json
      data/*.mcap
    episode_0001/
      ...
```

可以修改`review.local.json`中的`data_root`，或用命令保存路径配置：

```bash
python review.py --data-root "D:/robot_data" --save-config --list
python review.py --choose
# 非交互生成；--task是相对于数据根目录的路径
python review.py --task "blue_block_v2" --title "把蓝色积木放进杯子"
```

没有本机配置时默认读取`raw_data/`。数据根目录是任务文件夹的上一级；支持任务目录内再嵌套一层同名目录。`--task-id`可固定任务标识；`--criterion "标准"`可重复指定自定义审核标准。

`--generate-only`只生成检查和回放；`--no-browser`只启动服务；`--port 8777`指定其他端口。完整参数见`python review.py --help`。

## 如何审核

- 左侧搜索片段编号或按状态筛选，中间播放同步的左右腕视角，右侧填写备注与决定。
- **A保留 / R排除 / U待确认**；←/→切换。输入备注时不会触发快捷键。
- 决定自动保存到`tasks/<任务标识>/decisions.json`，也在浏览器暂存；可导出备份。
- “审核记录”导出`<任务标识>_review.json`，包含全部决定、备注和源文件指纹。
- “保留清单”导出`<任务标识>_approved.json`，只包含人工保留且技术未失败的episode。
- 两个JSON保留原有schema_version=2和相对路径语义，可交给ACT_train_new/Pi-train-new预处理。训练时仍需指向原始数据根目录，并保持相对路径结构一致。
- 待审核或待确认仍计入pending_count；不要把尚未完成的审核清单当作最终训练数据。

已有任务页下方可导入旧审核JSON；任务与文件指纹不匹配时拒绝导入。技术检查失败的片段不能直接保留。技术提醒和左臂0.1rad视角偏移提示不等于任务失败或安全限位。

## 已有审核产物的迁移

旧任务目录可以整体复制导入，不会移动源文件或覆盖同名任务：

```bash
python review.py --import-task "D:/old_review/tasks/task_id" --generate-only
```

导入的是检查报告、回放和审核记录，不是原始MCAP。重新生成相同任务会校验原文件指纹：内容未变则复用缓存，源数据变化则重新检查，并把旧决定归档到history。若需要承接旧审核记录，请保留原task-id。

## 目录结构

```text
data_review/
├─ backend/                 后端Python代码
│  ├─ server.py             任务管理、HTTP/API、审核保存
│  ├─ audit.py              技术检查与回放生成
│  ├─ nedf3_reader.py        MCAP/protobuf读取
│  └─ tests/                后端检查
├─ frontend/                前端HTML、CSS与JavaScript
│  ├─ index.html            任务总览
│  ├─ review.html           片段审核页
│  ├─ home.js               任务列表交互
│  ├─ app.js                审核、播放与导入导出
│  └─ styles.css            共用样式
├─ scripts/run_python.cmd   Windows解释器选择
├─ review.py                启动入口，原有命令保持有效
├─ start_review.cmd         打开已有任务
├─ prepare_task.cmd         选择并生成新任务
├─ start_review.sh          Linux入口
├─ requirements.txt
├─ review.local.json        本机数据路径与端口
├─ raw_data/                默认原始数据位置
└─ tasks/                   回放、检查结果和人工审核记录
```

前端通过 `/api/catalog` 和 `/api/decisions/<任务标识>` 与后端交互。后端同时提供静态页面，仍只需启动一个Python进程；无需另装Node或前端构建工具。前端资源位于 `/frontend/`，原任务地址和旧资源地址保留兼容。

后端检查：`python -m unittest discover -s backend/tests -v`。

回放为15fps压缩视频，仅用于审核；训练继续读取原始MCAP。工具只监听本机，不上传数据，不启动训练或机器人。


## 本次验收（2026-11-06）

- 审核、缓存、文件保护、HTTP保存等12项检查：首次11项通过；修复Windows短路径误判后，失败项复验通过，未重复运行其他已通过项。
- 一条真实MCAP用独立读取器完成CRC/解码/时间对齐检查并生成275帧回放，源文件指纹前后一致。
- 浏览器确认任务页、编号搜索、片段切换、倍速和视频播放正常，未出现页面脚本错误。
- 迁移的50条片段保持原审核记录字节内容和数据集指纹；本次界面检查未替用户作审核决定。

## 2026-11-06 目录整理

将代码分为backend和frontend，启动命令、配置位置、任务数据路径与审核JSON格式保持兼容。
