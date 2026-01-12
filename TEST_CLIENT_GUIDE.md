# test_ocr_client.py 新增功能说明

## 新增功能

在 `test_ocr_client.py` GUI 客户端中添加了两个新按钮，可以同时控制 OCR 识别和图像对比。

## 使用方法

### 启动程序

```bash
# GUI 模式（默认）
python test_ocr_client.py

# CLI 模式
python test_ocr_client.py --cli
```

### GUI 界面新增控件

在原有的按钮区域右侧，新增了以下控件：

1. **PointID 输入框**：设置图像对比的点 ID（默认 123）
2. **启动 OCR+图像对比 按钮**：同时启动 OCR 识别和图像对比
3. **停止 OCR+图像对比 按钮**：同时停止 OCR 识别和图像对比

## 功能说明

### 启动 OCR+图像对比

点击"启动 OCR+图像对比"按钮后：

1. **OCR 识别**：自动启动 ONLINE 请求的 Watch 模式
   - 持续发送 ONLINE 请求
   - 实时获取 OCR 识别结果（A/B/SkinDepth 等）
   - 请求间隔：使用 Watch interval 设置（默认 0.5 秒）

2. **图像对比**：发送一次 OFFLINE 请求启动图像对比
   - 使用 PointID 输入框中的值
   - 使用默认超时时间（100 秒）
   - 图像对比会自动持续监控治疗前后的图像

### 停止 OCR+图像对比

点击"停止 OCR+图像对比"按钮后：

1. **OCR 识别**：停止 ONLINE 请求的 Watch 模式
2. **图像对比**：发送相同的 OFFLINE 请求（相同 PointID）来停止图像对比

## 工作流程示例

### 典型使用场景

```
1. 启动服务器：python main.py
2. 启动客户端：python test_ocr_client.py
3. 设置 PointID（例如：456）
4. 点击"启动 OCR+图像对比"
5. 观察日志输出：
   - "已启动 OCR 识别（ONLINE Watch）"
   - "已启动图像对比（OFFLINE point_id=456, time_out=100）"
6. 进行需要记录的操作（治疗过程）
7. 点击"停止 OCR+图像对比"
8. 观察日志输出：
   - "已停止 OCR 识别（ONLINE Watch）"
   - "已停止图像对比（OFFLINE point_id=456）"
```

## 日志输出示例

### 启动时的日志
```
[14:30:15] INFO: 已启动 OCR 识别（ONLINE Watch）
[14:30:15] INFO: 已启动图像对比（OFFLINE point_id=123, time_out=100）
[14:30:15] WARN: OFFLINE 启动：无响应（正常）
[14:30:16] OK: {"SkinDepth": 5.2, "A": 4.3, "B": 3.1, "Alpha": 0, ...}
```

### 停止时的日志
```
[14:35:20] INFO: 已停止 OCR 识别（ONLINE Watch）
[14:35:20] INFO: 已停止图像对比（OFFLINE point_id=123）
[14:35:20] WARN: OFFLINE 停止：无响应（正常）
```

## 技术细节

### 服务器端逻辑

- **ONLINE 请求**：获取实时 OCR 识别结果
  - 返回：`{"SkinDepth": x, "A": y, "B": z, "Alpha": w, "Depth": v, ...}`

- **OFFLINE 请求**：启动或停止图像对比
  - 启动：发送新的 PointID
  - 停止：发送相同的 PointID
  - 参数：`{"point_id": 123, "time_out": 100, "is_save": true}`

### 客户端实现

- 使用线程发送 OFFLINE 请求，避免阻塞 GUI
- Watch 模式持续发送 ONLINE 请求
- 所有操作都记录在日志窗口中

## 注意事项

1. **PointID 的重要性**：
   - 启动和停止必须使用相同的 PointID 才能正确停止图像对比
   - 每次 PointID 不同都会启动新的对比任务

2. **响应行为**：
   - OFFLINE 请求可能没有响应（服务器端设计）
   - 这是正常行为，不影响功能

3. **Watch 间隔**：
   - 使用"Watch interval(s)"输入框设置
   - 默认 0.5 秒，可根据需要调整

4. **与原按钮的区别**：
   - "Start Watch"：只启动 OCR 识别
   - "Preset: OFFLINE"：只发送一次 OFFLINE 请求
   - 新按钮：同时启动/停止 OCR 和图像对比

## 界面布局

```
+--------+--------+-----------+-------+----------+----------+---------+----+----------+-------------------+----------------------+
|Send    |Start   |Stop Watch |-------|Preset:   |Preset:   |Preset:  |----|PointID:  |[123]             |启动 OCR+图像对比     |
|Once    |Watch   |           |       |ONLINE    |OFFLINE   |CLOSE    |    |          |                  |                      |
+--------+--------+-----------+-------+----------+----------+---------+----+----------+-------------------+----------------------+
```

## 常见问题

### Q: 点击启动后没有反应？
A: 检查服务器是否正在运行（`python main.py`）

### Q: 如何同时运行多个对比任务？
A: 修改 PointID，然后再次点击"启动"按钮。不同的 PointID 会启动独立的对比任务。

### Q: 为什么 OFFLINE 请求显示"无响应"？
A: 这是服务器端的设计（`server.py:248` 设置 `response = None`），不影响功能。

### Q: 可以只启动 OCR 识别吗？
A: 可以，使用原来的"Start Watch"按钮。

### Q: 可以只启动图像对比吗？
A: 可以，使用"Preset: OFFLINE"然后点击"Send Once"。

## 兼容性

- 完全兼容原有的 GUI 功能
- 新功能不影响原有的按钮和操作
- 可以混合使用新旧功能

## 更新日期

2025-01-09
