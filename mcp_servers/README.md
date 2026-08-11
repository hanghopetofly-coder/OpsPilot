# MCP Servers

本目录提供两个用于 AIOps 演示和离线测试的确定性 Mock MCP Server。它们不访问真实
CLS、Prometheus 或云监控，也不能直接用于生产诊断。

## 服务与工具

### CLS Server

- 地址：`http://127.0.0.1:8003/mcp`
- 传输：`streamable-http`
- 工具：
  - `get_current_timestamp()`：返回固定参考时刻的 Unix 毫秒时间戳。
  - `get_region_code_by_name(region_name)`：查询固定 Mock 地域映射。
  - `get_topic_info_by_name(topic_name, region_code=None)`：按名称查询 Mock 日志主题。
  - `search_topic_by_service_name(service_name, region_code=None, fuzzy=True)`：按服务查主题。
  - `search_log(topic_id, start_time, end_time, query=None, limit=100,
    service_name=None, scenario="normal")`：返回有界代表日志和重复错误聚合。

`search_log` 的时间参数是 Unix 毫秒；`limit` 必须为 1–100。无论重复次数多少，响应
最多返回 12 条代表记录，并通过 `statistics.repeated_errors` 保留聚合计数。查询支持
`level:ERROR`、`message:timeout` 或普通文本匹配。

### Monitor Server

- 地址：`http://127.0.0.1:8004/mcp`
- 传输：`streamable-http`
- 工具：
  - `query_cpu_metrics(service_name, start_time=None, end_time=None, interval="1m",
    scenario="normal")`
  - `query_memory_metrics(service_name, start_time=None, end_time=None, interval="1m",
    scenario="normal")`

时间格式为 `YYYY-MM-DD HH:MM:SS`；`interval` 只接受正整数分钟或小时（如 `1m`、
`5m`、`1h`），最大为 `24h`。单次响应最多 288 个点，包含统计值、异常区间、趋势和
阈值判定。

## 确定性场景

两个 Server 共用 `scenarios.py`，不读取墙上时钟、不使用随机数。省略或留空
`scenario` 时固定为 `normal`；支持：

| 场景 | 主要信号 |
| --- | --- |
| `normal` | 正常指标和健康日志 |
| `cpu_saturation` | CPU 升至饱和并出现 worker/队列错误 |
| `memory_pressure` | 内存压力、OOM 和 GC 信号 |
| `database_timeout` | 数据库查询超时与连接池耗尽 |
| `downstream_timeout` | 下游服务超时与熔断 |
| `conflicting_evidence` | 正常 CPU 指标与 CPU 饱和日志冲突 |
| `tool_unavailable` | Monitor 返回可分类的连接失败；CLS 保持可用作为替代来源 |

默认指标窗口固定为 `2026-02-14 10:00:00` 至 `2026-02-14 11:00:00`，而不是
“当前一小时”。`get_current_timestamp()` 将 `2026-02-14 11:00:00 UTC` 作为固定
参考时刻。调用方传入的合法窗口会被保留；起止相同的单点故障窗口采样场景峰值。

## 启动

先在项目根目录安装依赖，然后分别启动：

```bash
uv sync
uv run python mcp_servers/cls_server.py
uv run python mcp_servers/monitor_server.py
```

Linux/macOS 也可使用仓库现有 Makefile：

```bash
make start-cls
make start-monitor
make status-mcp
make stop-cls
make stop-monitor
```

`make start` / `make stop` 会连同 FastAPI 一起启动或停止所有服务。Windows 可在激活
`.venv` 后直接运行上面的 Python 入口；Makefile 的进程管理命令依赖 Unix 工具。

## 验证

这些测试不会启动网络服务：

```bash
uv run pytest tests/scenarios
uv run python -m evaluation.run
```

`tests/scenarios/` 校验时间与输出边界、重复性、场景信号和失败协议；
`evaluation/` 在固定 Raw Tool Result 上复用生产 Evidence、Ranking 和 Report 逻辑。

## 生产接入注意

替换为真实数据源时，至少需要实现鉴权、租户隔离、请求超时、服务端分页/限流、敏感
信息过滤、审计和稳定的错误分类，并保持当前有界输出契约。Mock 评测结果不能代表真实
模型、网络、CLS/监控后端或 Milvus 的质量。

更多应用配置和 AIOps 数据流见[主项目 README](../README.md)。
