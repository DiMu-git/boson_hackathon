# Higgs Audio Hackathon 实验计划（中文）

## 🎯 实验目标
研究在使用 **Higgs Audio v2** 模型进行语音生成时，不同 **prompt 设计**、**system prompt**、**场景描述 (scene)**、**speaker 标签** 以及 **解码参数 (temperature, top_p, top_k)** 对以下性能指标的影响：

- 声纹一致性（speaker embedding similarity）
- 语音可懂度（intelligibility / WER）
- 生成稳定性与延迟（latency、pitch、spectral smoothness）

---

## 🧪 实验假设
1. 添加明确的 `[SPEAKER*]` 标签能提高生成音频与参考音频之间的声纹相似度。  
2. 强化的 system prompt（如“Preserve the speaker’s identity”）能进一步稳定声纹。  
3. 不同场景描述（安静室内 vs 咖啡馆）会引入声学差异，从而影响 embedding 相似度。  
4. 不同风格的文本内容（正式、对话、情绪化等）会影响生成音的音色、韵律和声纹特征。  
5. 降低生成随机性（temperature、top_p、top_k）可减少声纹漂移。  
6. 使用多段参考语音比单段参考语音更能保持身份一致性。

---

## ⚙️ 实验变量设计

| 类别 | 参数代号 | 示例取值 |
|------|-----------|-----------|
| **System Prompt** | S0: 无系统提示 <br> S1: 默认提示 <br> S2: 加入声纹保持指令 | 共 3 种 |
| **Scene 描述** | D1: 安静房间 <br> D2: 咖啡馆 <br> D3: 小办公室 | 共 3 种 |
| **内容风格（文本）** | 中性、对话、公告、正式、平静、激动、悲伤、自信、快速、缓慢 | 共 10 种 |
| **解码参数组合** | (T, P, K) = (0.7,0.9,20), (0.5,0.85,10), (1.0,0.95,50), (1.2,0.98,100) | 共 4 种 |
| **Speaker 标签** | 无 `[SPEAKER*]` / 有 `[SPEAKER1]` | 共 2 种 |
| **参考音频数量** | 单段 (R1) / 三段 (R3) | 共 2 种 |

> ✅ 最初推荐运行组合：  
> - **Baseline Block:** S1 + D1 + T1 + R1 + K3  
> - 后续再做各参数 Ablation（如不同 system、scene、tag 等）

---

## 📊 输出与评估指标

| 指标 | 含义 |
|------|------|
| `sim_ecapa` | 生成音与参考音的 ECAPA-TDNN embedding 相似度 |
| `sim_feature` | 基于特征的相似度（pitch / spectral / MFCC 加权得到 `overall_similarity`）|
| `wer` | Whisper-Large 自动识别的词错误率（越低越好） |
| `latency_s` | 每次生成的 API 延迟（秒） |
| `pitch_diff`、`spectral_diff` | 声学统计变化指标 |
| `prompt_text` | 实际使用的文本内容 |

当前实现的“拆分评估（split evaluation）”会同时记录两组对比：
- `real_vs_real`（10% 真声 vs 90% 真声）：`mean_overall_real_vs_real`、`mean_embed_real_vs_real`
- `clone_vs_real`（克隆声 vs 90% 真声）：`mean_overall_clone_vs_real`、`mean_embed_clone_vs_real`

---

## 🧩 实验流程（与当前实现对齐）

1. 选择数据集说话人（支持固定 `--speaker-id`，如 211）。
2. 将该说话人的音频按 1:9 划分：10% 作为参考集（reference），90% 作为评估集（evaluation）。
3. 对每条 prompt：
   - 使用参考集（10%）和该 prompt 生成克隆音频（可控 `--max-train-clones`）。
   - 计算两组相似度：
     - 10% 真声 vs 90% 真声（real_vs_real）：特征相似度与 ECAPA 相似度；
     - 克隆声 vs 90% 真声（clone_vs_real）：特征相似度与 ECAPA 相似度。
   - 可用 `--eval-downsample-frac` 对 90% 集合下采样以加速评估（如 0.1）。
4. 将每条 prompt 的评估结果追加到统一 CSV，便于后续分析和可视化。
6. **绘制可视化结果**：  
   - 条形图：不同 prompt 下的平均相似度；  
   - 散点图：声纹相似度 vs 可懂度；  
   - t-SNE / PCA：embedding 在声纹空间的分布。

---

## 📁 输出文件结构（当前实现）
```
src/
  └── clone-audio/
      ├── <prompt_stem>/
      │   ├── clone_0000.wav
      │   ├── clone_0001.wav
      │   └── baseline_results.json  # 含 real_vs_real / clone_vs_real 各种均值指标
      └── ...

experiments/
  └── prompt_sweep/
      └── results.csv  # 汇总每条 prompt 的 split 评估结果
```

（可选）另一条“配置网格”路径 `scripts/run_experiments.py` 会执行基于 `experiment_config.json` 的基线块（S1+D1+T1+R1+K3），逐条文本合成并计算相对于首个参考的 ECAPA 相似度，结果写入 `experiments/higgs_prompt_eval.csv`。该路径不做 10%/90% 拆分评估，更多用于 API 级参数/场景的快速扫描。

## 🏃 运行命令（split evaluation / prompt sweep）

1. 生成 100 条文本内容：
```bash
uv run python /Users/mudi/Documents/boson_ha/boson_hackathon/tools/generate_prompt_texts.py
```

2. 设置鉴权并运行按 prompt 的拆分评估（固定说话人 211，10%/90%，下采样 10%）：
```bash
export BOSON_API_KEY="..." && export BOSON_BASE_URL="https://hackathon.boson.ai/v1"
uv run python /Users/mudi/Documents/boson_ha/boson_hackathon/scripts/batch_prompt_eval.py \
  --dataset librispeech --speaker-id 211 \
  --min-clips 40 --seed 42 --train-frac 0.10 \
  --sample-rate 24000 --max-train-clones 3 --eval-downsample-frac 0.10 \
  --prompts-glob "/Users/mudi/Documents/boson_ha/boson_hackathon/src/prompts/auto/text_*.txt"
```

3. 结果查看：
- 单条 prompt 结果：`src/clone-audio/<prompt_stem>/baseline_results.json`
- 汇总 CSV：`experiments/prompt_sweep/results.csv`

---

## 🧠 扩展实验方向

### 1. 跨语言声纹保持
测试相同 speaker 在中英文等不同语言输入下的 embedding 稳定性。

### 2. 自动 Prompt 搜索
使用 LLM 自动生成 100 条 prompt，计算每条的声纹相似度，找出最优语句特征（句型、情绪、语气）。

### 3. 声纹安全性评估
在 speaker verification / anti-spoof 模型上测试克隆音是否容易通过验证，从而研究模型安全边界。

---

## 🧮 分析方法
- 统计每组参数下的平均 `sim_ecapa`、`wer`、`latency`；  
- 对比分析：  
  - `[SPEAKER1]` vs 无标签；  
  - 单段 vs 多段参考；  
  - 不同场景描述；  
  - 不同风格文本；  
  - 不同温度与 top-k 控制。  
- 使用 t-SNE 或 PCA 可视化声纹 embedding 空间的聚类变化。

---

## 🏁 预期目标
1. 找出 **最能保持声纹一致性** 的 system + decoding 组合；  
2. 找出 **生成自然度与可懂度平衡最佳** 的参数；  
3. 为 Higgs Audio 模型提供声纹稳定性优化建议；  
4. 为 hackathon 提供创新性、安全性方向的研究成果。

---

## 📘 备注
- 实验运行需批量调用 Higgs API；  
- 需确保本地缓存生成的音频文件及指标；  
- 建议使用多线程或异步调度优化执行效率；  
- 若时间有限，可优先执行 baseline + 2–3 组 ablation。
