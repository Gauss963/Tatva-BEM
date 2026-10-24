# 避免反轉剛體模態的界面消去

2026-10-24。材料尺度測試引出一個低頻數值問題：動態 compliance 形式若直接
延伸到靜態，會反轉有限體的剛體零模態。新增獨立 Schur 實驗，保留相同支承、
Galerkin stiffness 和界面座標，只改變消去順序。
目前完成代數、靜態、複數頻率與短程摩擦／AD 檢查，沒有切換正式 backend，
完整 45 ms 接觸軌跡仍待驗證。
[結果與來源指紋](schur_condensation_results.json) 保留原式的病態與新式的殘差。

## 固定界面位移後的平衡

令 K 為兩個 block 經支承／platen 約束後的 nodal-force stiffness，
B 將其自由度映到 normal/tangential jumps 與 platen 位移。
原式利用各 body 的 compliance，最後反轉 `B K^-1 B.T`。
在 Re(s)>0 且條件良好時這是正確的；靜態 free-body K 含剛體模態，不能直接求逆。

一次 QR 分解建立固定 lifting P 與 nullspace N，使 `B P=I`、`B N=0`。
在指定界面位移 q 時，解未保留方向的平衡：

```text
A = N.T K N
U = P - N solve(A, N.T K P)
u = U q
Z = P.T K U.
```

因此 `B U=I`，且 `K U=B.T Z`。複數頻率使用 bilinear transpose，
不把它改成 conjugate transpose；沒有額外彈簧、阻尼、模態 clipping 或特徵值修補。
P、N 不隨頻率切換，亦未改變 CQ contour 或積分規則。
這是直接的受約束 Schur 消去，不是新引用的 BEM 複雜度定理。

## 實際 finite-block 檢查

|案例與頻率|原 supported K 條件數|消去矩陣 A 條件數|新／原 Z 相對差|
|---|---:|---:|---:|
|h80，s=1 /s|2.51e9|22.6|8.01e-9|
|h80，實際 2 us contour 的實軸端點|8.14e4|22.6|2.64e-13|
|fine2，s=1 /s|9.86e10|304|5.11e-8|
|fine2，實際 2 us contour 的實軸端點|3.20e6|304|1.59e-12|
|fine2，s=200+2000i /s|3.50e4|311|1.88e-14|

fine2 為 h20、contact5、end/corner0.5 mm 與 seed 局部加密，275 retained DOFs，
906 supported DOFs；新 A 有 631 DOFs。contour 對應 45 ms、22500 steps、FFT size
65536，端點為 s=175.6732939 /s。此處相對差使用 traction row scaling，
不是完整時間域誤差。

刻意靠近靜態的 s=1e-3 /s 時，fine2 原 body inverse 已接近浮點奇異；
與新式的差異很大，不能再將原 compliance 當成參考答案。
新式在靜態及所有取樣點仍核對完整 supported 方程與 retained jumps；
fine2 的最大相對平衡殘差約在 1e-15 等級，並保持 reciprocity。
真正使用的 CQ 端點則兩式吻合至約 1e-12，故目前沒有證據把約 25% 的 FEM 滑移差
歸因於此低頻消去問題。

另以獨立材料重新組裝驗證靜態 E 乘 2.5 時 Z 同比例改變；
動態 reconstruction 與原式一致，靜態 rigid 方向未被人工固定。
測試檢查全部 supported equilibrium、`B U=I`、工作形式的正性及 complex reciprocity。
這些有限點檢查不構成全頻率或非線性摩擦穩定性證明。
消去後條件數改善亦不直接等於整體 wall time 加速；診斷與其他工作重疊。

## 對 AD 的意義與剩餘驗證

固定 B 與材料參數時，可在 stationary lift 上寫
`dZ=U_left.T (dK) U_right`；對稱 K 時左右 lift 相同。
這可避免反傳每次線性迭代與 QR，但 `dK`、static lift 和時間歷史的導數仍需接通。
這是後續材料 AD 的推導方向，尚未列為已實作功能。
幾何變動會同時改變 B、支承座標與積分權重，不能只套這個固定 B 公式。

## 兩階段 trace 消去與短程整合

另實作先固定每個 body 的接觸／platen trace，再處理共同 trace 運動的兩階段形式。
fine2 的 trace 有 548 個 DOFs：分別消去兩個 body 的 198／160 個內部自由度，
最後消去 273 個共同 trace 自由度。全部被消去的方向與原 631 維 nullspace 相同，
但不需要一次分解 631 維矩陣。
在同一實際 contour 端點，三個待解 block 的條件數分別為 217、79、89，
與原 compliance 的 Z 相對差仍為 1.59e-12。這不包含 V/H/D 組裝的成本，
不能由這些維度直接推算整體加速倍數。

新 trace operator 以 h80、Q12、4 us／64 步、actuator 增量 2 mm 的加速診斷
直接建立 native resolvent，再接回原 JAX 接觸 solver。
相對獨立 dense Z 解，壓力最大差 4.06e-9 MPa、剪力 3.64e-9 MPa、累積滑移
2.59e-11 mm；七個欄位通過原 1e-5 絕對差 guard。
解代回原完整 Z 歷史，含初始誤差的 residual 界為 4.49e-9 MPa，小於原 1e-7 容差。
所有滑動、開閉、weakening clip 及 clip 等值分支相同。
五參數條件式 adjoint 最大相對差 1.72e-9；actuator cotangent 最大絕對差 5.56e-12，
初始 z／c cotangent 差分別小於 9.62e-11／6.29e-10，通過原容差。
這是新消去形式的整合核對，不是原 45 ms 載入或 FEM 物理精度驗證。

四個 frequency workers 共用唯讀 trace 基底的測試，亦與序列結果逐位元相同。
prepared kernel 與 native 頻譜保存實際 formulation、basis 尺寸及 source hashes；
construction estimate 額外計入 QR／投影／recovery 工作陣列。
共用的 `operator_resolvent.py` 提供私有 factory binding，不改 numerical package globals。
現行混合積分長計算的 wrapper 保持原版本，待它完成後才適合整理到共用入口。

下一步仍須核對實際整個 contour、完整接觸分支與條件式 adjoint，
正式 45 ms 比較沿用同一物理參數與原驗收門檻。

```sh
FDBEM_CORES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=bem/src python -B \
  bem/tools/audit_schur_condensation.py SOURCE_RUN \
  --dt 2e-6 --duration .045 --max-memory-mb 4096 --output NEW_AUDIT

# 同一 auditor 可用 --formulation trace 檢查兩階段形式。
# 直接建立 trace Schur kernel；其餘參數沿用 native producer。
FDBEM_CORES=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=bem/src python -B \
  bem/tools/prepare_schur_resolvent.py SOURCE_RUN --formulation trace \
  --workers 4 --max-memory-mb 4096 --output NEW_KERNEL
bem/run.sh resolvent-simulate NEW_KERNEL --fft-cache-mb 8 \
  --max-memory-mb 4096 --output NEW_RUN
```

SOURCE_RUN 決定原幾何和積分階數。預設取樣包含靜態、接近靜態、複數波頻及
從指定時間網格算出的真正 CQ 端點；不配置完整 CQ 時間權重。
第二個命令預設沿用 SOURCE_RUN 的時間網格與完整 horizon，
也可明確給 `--dt`／`--duration`。完成 kernel 仍需另跑原方程、分支與 AD 比較。
