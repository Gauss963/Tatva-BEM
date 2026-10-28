# 內部位移、速度、應力重建與獨立波傳控制

2026-10-28 新增 [`interior_fields.py`](../tools/interior_fields.py)、
[`interior_radau.py`](../tools/interior_radau.py) 與
[`interior_radau_adjoint.py`](../tools/interior_radau_adjoint.py)。
它們是獨立工具，沒有修改正在執行的 FEM、Radau 建核或接觸程式。
本輪補上內部場算子與解析控制，完整 event-16 的內部場比較仍未完成。

## 表示式與離散量

再核對 Aimi、Di Credico、Gimperlein (2024)
[§5.1、式 (46)–(50)](https://arxiv.org/html/2405.08566v1#S5.SS1)：
域內位移由單層與雙層位勢重建。連續的對稱與非對稱 DtN 表示相等，
離散後則不相等。因此目前 P1/P0 對稱 Galerkin 的節點力 `S u`，
不能除以節點權重後當作 P0 traction density。
工具用同一個原始 Galerkin `V lambda = H u` 恢復輔助密度，再計算

```text
U(x,s) = Vp(s) lambda(s) - Kp(s) u_boundary(s)
d_k U = -integral(d_k G lambda) + integral(d_k D u_boundary)
sigma = lambda_Lame tr(grad U) I + mu (grad U + grad U.T)
sigma_zz = lambda_Lame tr(grad U)       # plane strain
```

分離向量維持 `source - target`。應力使用解析的 Green Hessian，
沒有對位移結果作空間差分；小 Bessel arguments 在微分前用 power/log series。
靠近邊界的積分按 target 距離細分面板；邊界、外部點與超過細分上限的情形被拒絕，
不把邊界 jump term 加進域內表示式。

`InteriorRetainedTransfer` 使用與原 solver 相同的兩物體支持與
`q=(normal jump, tangent jump, platen)`。每個頻率的原始 V/H/D 只組裝一次，
共用於全邊界恢復、密度與所有 probes。小型兩物體控制已核對 impedance 與
boundary reconstruction 均符合原 `NodalInterfaceTransfer`，另與直接密度恢復核對。
這是代數控制，不能取代對連續問題的收斂測試。

靜態 lift 必須另外重建，再加零初態的動態擾動；只有擾動速度才用 `s U(s)`。
不能把非零初始位移放入零初態 CQ，也不能因後處理而抹去原 solver 保留的 r0 修正。

## 解析 P/S 波的實際結果

[完整結果、來源與資料 SHA256](interior_fields_results.json)。
一致單位的 unit square、`E=3, nu=.23, rho=1.2`，不是 PMMA event-16。
空間控制使用 `s=1.4+0.9i` 的解析 P/S 波、P1 位移與 P0 密度資料；
`h=.125→.0625` 時，位移／梯度／應力差約縮至四分之一。
另有靜態 affine patch，包括距邊界 `1e-5` 的點，並驗證錯用 lumped nodal force 會失敗。

時間控制輸入支撐於 `[.2,.6]` 的 C5 pulse，直接對照已知延遲的解析場，
沒有擬合相位。用完整 rectangular Radau CQ，保留所有所需 lag 與 finite-DFT alias 項。
以下為**所有存檔 endpoint 的相對 L2 差**，不是最大點誤差、未存檔時間界或穩定性證明。

| 波 | h | dt | 位移 | 速度 | 應力 |
|---|---:|---:|---:|---:|---:|
| P | .03125 | .04 | 4.0837% | 10.6278% | 10.6301% |
| P | .03125 | .02 | .6455% | 1.7357% | 1.7146% |
| P | .03125 | .01 | .2391% | .4899% | .3680% |
| P | .015625 | .01 | .0971% | .2513% | .2391% |
| P | .015625 | .005 | .0579% | .1128% | .0766% |
| S | .03125 | .04 | 6.4941% | 16.1272% | 16.0954% |
| S | .03125 | .02 | 1.1875% | 2.9622% | 3.0845% |
| S | .03125 | .01 | .4664% | .9796% | 1.4026% |
| S | .015625 | .01 | .1912% | .4589% | .5199% |
| S | .015625 | .005 | .1102% | .2278% | .3445% |

這組控制分開展示時間與空間誤差：固定 h 縮小 dt 後仍有空間誤差，
不能只看 Newton residual 或單一時間收斂率。它使用匹配平面波的邊界資料，
沒有驗證有限體反射、摩擦開閉、全域物理能量或 FEM 一致性。

## AD 與實作範圍

內部觀測對 retained stages 的固定核心映射是線性的。
`interior_pullback` 回傳所有實際 input stages 的 cotangent，包含完整的未來依賴，
不需要保存 forward trajectory 或對頻率分解／求解的迭代過程反傳。
人工的初始零 stage block 是固定資料，梯度為零。
完整點積恆等式通過 `1e-14` 相對容差；加權觀測 L2 objective 的 pullback
與獨立 JAX reverse gradient 通過 `1e-12` 容差。
這只驗證固定 kernel 的線性轉置，尚未接入完整 event 的初始化／接觸 adjoint。
材料、幾何與靜態 lift 的額外導數仍需另外處理，不能把此結果當成完整參數 AD。

rectangular CQ 目前是小型 dense 參考，組核 array budget 預設 512 MiB；
大於預算會拒絕，不刪除歷史或改成局部阻尼。caller state 與 BEM assembly 不在這個
array estimate 中。完整細網格後處理仍需要有界記憶體的執行器，不能直接以此 reference
執行整個 45 ms。23 項相關測試通過，包含解析 Hessian 與獨立差分、static limit、
P/S 空間控制、已知延遲的位移／速度時間收斂、retained mapping、完整歷史轉置與 JAX。

## 下一個完整驗證所需資料

目前 graded FEM 快照的內部 displacement／stress 是 float32，完整 bulk 只配置
69 frames（shear 64 frames），沒有高頻率的內部速度紀錄；5 µs 接觸輸出只涵蓋界面。
已直接檢查保留的 HDF5 dataset schema。因此該 reference 可用於新的接觸場對照，
不能據此宣稱已驗證高頻率內部波。後續完整波場比較需要額外的 true float64 interior
probe observations，並保持原積分／接觸方程不變。
這不更動或重啟正在進行的參考工作。

重現這組解析控制：

```sh
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -B bem/tools/check_interior_fields.py \
  --output bem/outputs/new-manufactured-interior-controls
```

本機驗證與解析控制的四個序列控制區間合計約 36.77 秒，包含在同一 FEM 程序的
wall cost，並已恢復同一 PID。這些是正確性控制，不是同精度速度基準。
