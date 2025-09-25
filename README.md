
# Basic AI Chatbot (Gradio + OpenAI)

## 快速开始
1. 创建并填写环境变量：
   - 新建 `.env` 文件，写入：
     ```bash
     OPENAI_API_KEY=你的OpenAI密钥
     ```
2. 安装依赖：
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. 运行（Task 1）：
   ```bash
   python task1_app.py
   ```
4. 运行（Task 2 - YouTube Chatbot）：
   ```bash
   python task2_app.py
   ```
5. 运行（Task 3 - 概念图可视化）：
   ```bash
   python task3_app.py
   ```

## 功能
- Task 1：
  - 多会话、清空当前会话、简洁UI
  - 短期记忆（每会话最近若干条）与基础错误处理
- Task 2（YouTube）：
  - 自动检测 YouTube URL，抓取转录（含常用语言与翻译回退）
  - 在聊天中展示视频信息（标题/频道/封面）与 3-4 句摘要
  - 基于转录进行问答，尽量引用时间戳（例如 [05:30]）
  - 会话级缓存
- Task 3（概念图）：
  - 软匹配指令（如“Show me the concept map”、“Visualize video concepts”等）
  - 基于转录抽取概念、构建语义关系图（NetworkX）
  - 用 Plotly 显示交互式概念图：
    - 节点大小代表重要性（基于中心性）
    - 边的粗细代表关系强度
    - 颜色区分概念类别（基于度数分档）
    - 悬停/点击显示细节

## 环境变量
- `OPENAI_API_KEY`: OpenAI API 密钥
- 可选 `PORT`: 启动端口，默认 7860

## 备注
- 默认使用 `gpt-4o-mini` 模型，可在各 `task*_app.py` 中调整。
- 首次使用 spaCy 可能需要下载语言模型：`python -m spacy download en_core_web_sm`。
