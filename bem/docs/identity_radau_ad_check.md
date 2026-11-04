# 完整 RK 導數檢查中的局部 AD 切線

`tools/check_identity_radau_ad.py` 是原完整 AD checker 的可選入口。
它保留原始 causal audit、primal SHA、靜態場重算與初始 sensitivity、
reverse、七方向 JVP、完整轉置歷史，以及原 `rtol=1e-7, atol=1e-18`。
僅將 identity jump maps 的切線組裝替換為已驗證的
[局部 JAX AD 與 nodal scatter](identity_stage_tangent.md)；一般 maps 使用原切線。
原工具與執行中的細網格排程皆未修改。

2026-11-04 的[完整 128 步雙 block 控制](identity_radau_ad_check_results.json)通過：
五參數梯度最大差 `1.9853e-23`，七個 JVP、全部 actuator cotangents、初始狀態／路徑
cotangents 與三組初始場 Jacobians 均逐位相同。
reverse 與三批 JVP 所建立的四個 system 全部實際使用新的切線組裝。

控制使用額外平均 0.1 核心，完整事件保留 3.8；
最終 `wait4` CPU 為 `20.318318 s`，含等待總時間 `203.193219 s`，
實際平均 `0.099995` 核心，全部 CPU 等待額度已償還。
這是與完整模擬並行的導數等價控制，其秒數不能作隔離效能比較。
先前單項細網格切線組裝的改善亦不能直接當作完整 AD 或模擬加速倍數。

使用與 kernel 相同的 x64／runtime 設定，並由外部四核心管理程序執行：

```bash
PYTHONPATH=bem/src python -B bem/tools/check_identity_radau_ad.py RUN \
  --operator-audit ORIGINAL_CAUSAL_AUDIT --output NEW_AD_DIRECTORY \
  --jvp-batch-size 3 --fft-cache-mb 8 --fft-rows 2 --max-memory-mb 12288
```

參數及其他檢查沿用 `check_compact_radau_ad.py`，預設包含 selected healed active-set 初始化導數。
目前驗證固定 bulk、接觸參數與已記錄的載入；沒有 material／geometry／endogenous stop-time 導數，
也尚未證明完整細網格事件的梯度收斂或同精度 FEM 優勢。
