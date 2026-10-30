# 遠場 ACA 組裝：精度控制通過，整體速度尚未改善

本控制把[精確 block／fiber 取值](galerkin_blocks.md)用於原始 V/H/D 的空間壓縮，
沒有修改 event-16 的物理參數、CQ、支承、初始種子或摩擦律，也沒有替換正在執行的原 producer。
方向參考 Keshava、Schanz [2026 預印本 §3.1](https://arxiv.org/html/2607.15292v1#S3.SS1)
的 near/far 分工；這是二維 scalar-cross 空間組裝原型，不是該文完整 3D-ACA／gCQ 的移植。

## 實作與限制

- 共用 tree 的每個 box 包含 P0 panel 與 P1 node 的 **support union**，並保留兩個向量分量。
  這個保守的 union 避免把 nodal support 誤當成 midpoint；它不等於分別最佳化兩種 basis 的 tree。
- 近場保留原 Q12 積分。遠場用 complex ACA，完整 rows 作 empirical holdout checks；
  高秩、停止／成本條件不合格時回到完整取值。這些檢查不是所有未取 entries 的嚴格誤差界。
- 同一面板對同時計算 V_rc、H_rc、D_rc、H_cr，V/D 保留 bilinear reciprocal transpose，
  H 的兩個方向各自計算。沒有假設 H 對稱、做共軛 transpose 或 eigenvalue clipping。
- 原始幾何 request 先按 quadrature kind 打包，以減少 one-pair batches；
  原積分點、moment／奇異修正不改，加總順序可能改變。
- 每次 frequency evaluation 有自己的 exact local-pair-value cache（每 body 約 66–67 MiB，
  單一 cache 上限 128 MiB）；不能跨 mesh、材料、reference scale 或頻率沿用。
  更大 mesh 超過 cache 上限時退回 uncached kernel；此上限不是整體 RSS 限制。
- 近場幾何可唯讀共用，每 body pinned array cap 512 MiB。
  最後仍產生 **dense V/H/D**，沿用原 global V solve、Schur elimination 與 full-boundary recovery；
  不是 H-matrix direct solver，也沒有壓縮完整頻率核或刪除時間歷史。

近場的已知 Frobenius norm 是完整場 norm 的下界，用來分配 far-block 的 mixed tolerance。
驗收保留 `1e-14 + 1e-11 * ||A_reference||_F`；內部 ACA 再乘 0.05 safety factor。
即使有這個 factor，仍不宣稱 uniform／certified error；本次另逐項組出完整原矩陣作獨立稽核。

19 項測試通過：support／partition 全覆蓋、paired H 方向、complex cross approximation、
完整 bilinear transpose、高秩 fallback、長方形 lookahead 終止、JSON metadata、
兩個 body 的完整原始層／stiffness、batch packing、同頻 cache 重用與跨頻拒絕、roundoff closure 的 support enclosure，及 cache cap fallback。
這尚未驗證完整非線性事件或 initialized contact AD；material／geometry AD 也未新增。

## 實際細網格控制

相同 event-16 mesh：outer 10 mm、contact 2 mm、end/corner 0.015625 mm、Q12，
531 retained inputs。每頻率兩次交替 original→hierarchical／hierarchical→original，
BLAS 單執行緒，原自有 BEM 暫停後才量測、完成後恢復。
兩個 body 皆比較完整 V/H/D、凝縮阻抗與完整邊界重建。

|2 µs／45 ms Radau 實際 s（1/s）|原雙 body 組裝中位數|packed/cached ACA 中位數|
|---|---:|---:|
|175.6733 + 0i|53.568 s|87.767 s|
|175.7813 + 31398.6686i|97.822 s|126.956 s|
|2999473.0418 + 0i|52.109 s|31.601 s|

原幾何準備 22.97 s；hierarchical 幾何準備 10.03 s，兩者都在完整控制成本中保留。
本控制連所有重複與獨立檢查共 941.36 s，外層連測試為 952.94 s。
只有高衰減實頻較快；**不能拿單點的約 39% 耗時下降代表整個 contour 或完整事件的加速**。

所有 raw mixed-norm gates 均通過，最大 normalized error 約 0.01997（門檻 1）。
兩個 Schur outputs 的最大 relative infinity norm 差約 1.55e-10。
三點 `Hermitian(Z(s)/s)` 的最小 eigenvalue 仍為正；擾動的 Frobenius upper bound
均遠小於原正定 margin。這是取樣點的線性 port 檢查，不是完整 CQ／摩擦接觸穩定性證明。

## 為何沒有全面採用

低／複頻僅約 10% 的 unique physical pairs 被省下；多個場的不同 pivots 與高精度
要求造成大量 exact fallback。以 moving body 的低實頻為例，140 個 far blocks 中
僅 32 個保留 ACA，108 個 fallback；仍評估 108601／120295 unique pairs。
幾何準備與零碎 request 的額外成本超過節省的 Bessel／quadrature 成本。

第一次未打包版本不只更慢（低實頻 121.41 s 對原 53.39 s），
第二個頻率的 stationary H/D raw normalized error 1.49／2.32，已拒絕。
雖然該次 Schur maps 通過，也沒有因此放寬 raw gate。
首次 plan 統計另有 NumPy scalar serialization 問題，發生在 frequency evaluation 前。
這些成本、失敗與原 source snapshots 均保留在[機器可讀結果](hierarchical_galerkin_results.json)，
沒有從 elapsed time 隱藏。

下一步應先處理重複幾何準備或 special-function 評估成本，並研究頻率維度壓縮；
不把目前 backend 整體接入昂貴的完整 producer。若後續選擇 hybrid backend，仍須
跨 contour 控制、完整 CQ／primal／initialized AD 與總成本驗證，才能判斷原目標。
FEM 的[破裂後時間收斂失敗](fem_postrupture_timestep.md)也仍是必要的參考解工作。
目前沒有相同精度下的完整 FEM／BEM 速度或穩定性優勢。

## 原始組裝的計時拆解

為避免把優化方向建立在推測上，另對相同 moving body 的原組裝加上 timer-only wrappers。
所有原數值 function bytecode 與運算呼叫保持原樣，原 module bindings 不改。
兩個頻率的完整 V/H/D 均與原版本 bitwise 相同；timer overhead 保留在成本中。

|原 moving body 頻率|profiled 組裝|Bessel K0/K1|小參數級數|主組裝 einsum|
|---|---:|---:|---:|---:|
|低實頻|26.434 s|5.542 s（21.0%）|3.099 s（11.7%）|7.988 s（30.2%）|
|175.7813 + 31398.6686i|49.849 s|31.809 s（63.8%）|0.166 s（0.33%）|7.978 s（16.0%）|

`radial_total` 包含 Bessel／級數，不能把兩者再次加進 total。
其餘可分開的 tensor／traction／scatter 與未歸類時間，也在結果 JSON 保存。
這是兩個頻率的 bottleneck profile，沒有拿 instrumentation timings 當成改善前後速度比。
它支持下一輪優先研究複頻 special-function 評估，以及低頻級數／收縮的代數成本；
任何新近似仍需原始矩陣、完整因果歷史與解／梯度的獨立控制。
