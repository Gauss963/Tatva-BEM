# 混合建核快取的滑動與初始化 AD 驗證

`check_mixed_radau_nonlinear.py` 現可讀取已完成的原版、原版 prefix → JAX 續跑快取，
直接驗證完整短程滑動與初始化梯度。這補上[建核／複製檢查](jax_radau_producer.md)
之後的非線性整合環節；沒有遷移正在執行的細網格快取，完整事件的 FEM 一致性與優勢仍未證明。

本次重用已付費的兩份 128-step kernel，共約 4.7 MB，而不是重新計算動態頻率矩陣。
原 pytest 暫存目錄的全部檔案逐位元複製到獨立輸入資料夾，source hashes、完整
weights、wrap、startup 及 lag-zero impedance 均重新核對。曾先選到舊 backend 的快取，
因來源 hash 不符而拒絕，沒有拿它進行混合模擬；該紀錄仍保留。

## 相同方程、實際保存權重

驗證沿用既有 `check_paired_ray_nonlinear.experiment` 的原靜態初始化、primal、
implicit-root AD 與完整轉置歷史，只在私有 AST namespace 中替換取得 kernel 的區段。
原程式、原 Tatva 與執行中的 producer 都不修改。

- 使用**實際保存的 physical time weights**，不以重新反轉出的權重代替它們。
- compact spectra 原為 `sqrt(w) L sqrt(w)`；還原 physical coordinates 後，
  另以完整 IFFT 核對保存權重與 wrap，門檻仍為 `rtol=1e-8, atol=1e-12`。
- 檢查原建核的 repo 數值來源：原版 15 個、混合版 20 個均與目前來源相同。
  來源版本、完整完成狀態、離散設定與 runtime 不符就拒絕。
- 兩個快取的 lag-zero impedance 要求逐位元相同。
- `dt、steps、DOF、FFT size、radius` 必須完全相同。disk／memory 的
  workspace estimate 可不同，不能把這個估計欄位誤當成數學 contour。
- 實驗的 transfer 與 prepared objects 被設為禁止動態組裝；意外呼叫就失敗。
  共享的原靜態 preload 仍由原靜態方法建立。

本案為 11 history DOF、5 個接觸點、Q12、2 µs × 128 步，
摩擦路徑保留 `(8,16)` 子步與所有歷史。
bulk 為原有限兩體 Galerkin 模型，起始條件使用相同 f09 healed seed，
保留量測的靜態力及非零殘差之完整歷史與初始化導數。

為了在短時間內進入弱化，載入是**診斷用 7812.5 mm/s**，不是原 event-16 的 20 mm/s。
由保存的 0.5 mm startup 線性換算為 2 mm，並以原完整 causal startup 公式獨立核對。
這利用固定 bulk 的線性，沒有縮短摩擦歷史、平移時間或改變摩擦律。

## 完整離散軌跡與 AD 結果

AD 在收斂的接觸根上解隱式線性化系統，沒有對每次 Newton iteration 反傳。
一次 reverse pass 配合完整轉置卷積歷史，取得五參數與所有 stage loading 的 cotangent；
七方向 forward pass 則在每一步共用同一 tangent matrix 的多 RHS solve。
初始化的位移、plastic offset、量測力與殘差也保留導數，避免只微分 transient、
卻把參數相依的起始平衡凍結。這延續原 AD 設計，本輪驗證其與混合快取接合後的一致性。

|對照項目|混合版相對原版差|
|---|---:|
|全部 causal stage weights|2.03e-14|
|全部保存的 stage 位移|3.82e-14|
|全部 stage 接觸 response|2.10e-13|
|五個接觸參數的 initialized gradient|1.01e-12|
|全部 stage actuator gradient|7.57e-14|
|七個 initialized JVP 結果|1.05e-12|

兩者最終最大累積滑移均約 **1.033821980 mm**，確實超過 critical slip，
不是全程黏著的零梯度控制。所有 128 步均完成，場有限、累積量單調、
接觸強度違反量為零，沒有 root recovery event。

五個接觸參數為 `kn, kt, mu_s, mu_k, Dc`；bulk 材料與幾何固定。
五個參數方向加兩個完整 stage loading 方向共七個 JVP，
對照一次 VJP 的五參數與全部載入 cotangent；本次原 engine 是**七方向同批**。
先前 JAX 報告將此短程控制標成 `3+3+1`，現依實際呼叫更正；
大型 checker 的可配置分批機制不代表這個控制曾採用它。

跨 backend 梯度與雙方 JVP/VJP duality 均通過原混合門檻
`rtol=1e-7, atol=1e-18`；最大 normalized error 約 **5.96e-4**，門檻為 1。
這驗證的是所選 healed-static／transient 分支的離散導數，
沒有獨立的完整軌跡 finite-difference 驗證，也沒有材料、幾何、內生 stopping time 導數或梯度的網格收斂證明。

## 來源與成本

六項輸入測試涵蓋 owned-copy path、實際保存權重逐位元不變、workspace estimate
差異、損壞 bytes、matching-hash 的非有限資料、改變數學 plan 與 runtime。
第一版因把 workspace estimate 一併比較而保守拒絕實際快取；
改用原 `cq_plan_signature` 後完整滑動／AD 通過，未更改任何物理精度門檻。
再加強原建核來源 guard 後，重新通過完整控制。

控制沿用平均 3.85 CPU quota，短暫暫停原 BEM，結束均恢復相同三個程序。
所有失敗與重核成本都保留；不能把重用既有 kernel 的約 12 s 控制當成建核加模擬的總耗時。
三輪控制及發佈的外層成本共 **33.804 s**，含第一輪拒絕與來源 guard 加強後的重核。
本次沒有新的 physical contour evaluation，因此結果明確標記
`all_physical_contour_modes_checked=false`，不沿用原直接組裝實驗的完整 contour 宣稱。

```sh
PYTHONPATH=bem/src JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 \
  python -B bem/tools/check_mixed_radau_nonlinear.py \
  /path/to/original-kernel /path/to/mixed-kernel --output /path/to/fresh-report
```

此工具刻意驗證上述固定短程控制；它不是原 event-16 的執行入口。
來源／runtime 凍結也代表不同環境須重新建立相符快取。
[完整數值、來源、失敗紀錄與成本](mixed_radau_ad_results.json)，以及
[原版](diagnostics/mixed_radau_original_arrays.npz)／[混合版](diagnostics/mixed_radau_mixed_arrays.npz)
的全部保存 stage 場、參數／actuator gradients 與七個 JVP 已保存。
原 `/Volumes/Gauss-T7/tatva` 維持唯讀；完整細網格 BEM、FEM 時空收斂與同精度優勢仍須完成。

前一輪 `8fa6744` 的[完整 Linux CI](https://github.com/Gauss963/Tatva-BEM/actions/runs/41270681970)
已在 Python 3.11、3.12 都通過，確認原起始阻抗修正通過跨平台測試。
