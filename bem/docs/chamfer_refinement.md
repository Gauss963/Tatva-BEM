# 保留原 chamfer 的 FEM 端點加密控制

相同 241 contact nodes 的 BEM 靜態解相對 FEM h2 仍有約 9.53% 壓力、11.94%
剪力 P1 L2 差；outer h10／h5 的差僅約 0.0224%／0.0596%。
因此大型 CQ 建核已停在原篩選門檻，沒有把較小的 root residual 當成精度通過。
後續先分開檢查 FEM／BEM 的 contact-end 空間誤差，而非以 mesh 調參擬合 endpoint peak。

原 FEM source `933ae1f` 的 `create_structured_quad_block` 明確不支援同時使用
chamfer 與 fault-end refinement。直接改成 fillet 會改變本事件幾何，不能沿用。
`tools/chamfer_mesh_refinement.py` 新增明確的 mesh-only adapter：

- 使用原 graded axis、quad connectivity 與 chamfer polygon mapping。
- 在兩個 body 共用的 y refinement points 加入真實接觸終點 `500-20=480 mm`。
- 複製 function globals 以選擇新的 mesh generator，不修改原 function bindings 或 source files。
- 保留原材料、Coulomb active set、penalties、支承與 loading kernels。
  原 RunConfig 的 graded pair-mass time-step estimate 仍由原 model builder 計算。

沒有 refinement 時，adapter 直接呼叫原 builder。三項測試使用從 pinned source
原樣擷取的 geometry functions，核對該路徑的 bitwise mesh identity、兩側接觸節點、
chamfer 面積、所有 Gauss-point Jacobians 為正，以及原函式仍保留原有拒絕條件。
test fixture 保存 source file／各函式 AST 指紋與 MIT 來源標示；不依賴 CI 中的本機 FEM cache。

`fem_graded_static.py` 使用這個 adapter 和原 static solver，從相同 captured float64
restart 的 P1 場重新解每個網格，容許原 healed Coulomb preload 更新 plastic offset。
其輸出明確在 shear-start 將 weakening memory 設回零；不把 static helper 回傳的
absolute plastic offset 當成動態累積弱化。

小型真實 FEM 控制已完成：outer h40、end h10，原／加密模型為 318／946 DOFs，
平衡 residual 約 1.11e-12／1.80e-12。這只驗證介面能接回真實 source；不是原事件的精度結果。
原 bulk h2 的控制已完成，保留 unrefined baseline，將 end size 從 1 mm
連續減半到 0.015625 mm。原 h2 baseline 的 pressure peak 為 344.6162 MPa；
最細控制為 1519.7601 MPa。最後兩級的全接觸面 pressure／shear P1 L2 變化為
1.8834%／0.3161%，仍不能視為連續體或完整動態已收斂。

```sh
PYTHONPATH=bem/src FDBEM_CORES=4 OPENBLAS_NUM_THREADS=1 JAX_ENABLE_X64=true \
  python -B bem/tools/fem_graded_static.py --restart-seed ORIGINAL_FLOAT64_SEED \
  --mesh-size 2 --end-sizes 1 .5 .25 .125 --include-unrefined --output FRESH_STATIC_STUDY
```

使用與既有研究相同的完整四核心 thread environment；上例只列主要變數。
工具保存輸入／adapter 指紋、effective RunConfig、DOFs、接觸 spacing、真實 float64
static snapshots、原方程 residual、active set、推估的 model dt、RSS 與計時。
自動 dt 是 model setup 資料；本工具沒有時間推進，也沒有驗證這個新 mesh adapter 的動態結果。
目前尚未將 adapter 接入正式 FEM transient runner；原 h2／h1 完整動態 reference 均保留。
原 `/Volumes/Gauss-T7/tatva` 全程只讀，同精度 BEM 優勢仍未達標。

## BEM 靜態比較與初始化的數值修正

BEM 使用 contact scale 2 mm、outer scale 10 mm，獨立減小端點與角落面板。
最初的 condensed backend 在 end 0.015625 mm 觸發既有 full-body balance 門檻：
重建位移 `u = loaded_u + response_u @ q` 的原始方程殘差為
1.9315e-7 MPa，超過 1e-7；trace error 約 1.12e-13 mm。
原始失敗、已完成的前六級控制及其 source commit `050bf1d` 均保留。

新增的 `tools/uncondensed_static_contact.py` 在原支承／platen 消去後的完整
boundary displacement coordinates 聯立求解 healed active set，保留原 penalty、
摩擦係數、source restart、weakening memory 及全部接受門檻。
它直接量測 `Kr*u - F`，以原 work-conjugate trace 的轉置投影取得 bulk port force，
並各自檢查 full balance、非 port 投影殘差及 port equilibrium；沒有把 bulk force
設成負的 contact force 來製造零殘差。

在完全相同的失敗矩陣上，直接 full-body root 的殘差為 1.17e-8 MPa；
與舊 trace 解的位移差約 1.21e-12 mm、完整位移差約 6.54e-12 mm。
這是原方程的數值初始化改善，不是整段動態穩定性證明。
三項新增測試包含原／refined quadrature 的場與功共軛平衡、受擾重建矩陣的拒絕，
以及直接求解不依賴受擾重建的檢查。

新的 BEM end sizes 0.0625、0.03125、0.015625、0.0078125 mm 全部通過原門檻；
最後兩級 pressure／shear 的全接觸面 P1 L2 變化為 0.9087%／0.1125%。
end 同為 0.015625 mm 時，對 FEM h2 的 pressure／shear 差為 0.9165%／0.6189%；
BEM 再細一級而 FEM 不變時，差為 1.2206%／0.5979%。全部指標包含端點，
不能挑較接近的一級作為收斂證書。

兩方法都從同一 float64 P1 source 取樣，但新網格沒有保留全部 source knots。
BEM 的固定 jump trial-shear projection bound 約 3.26–3.73 MPa，已列入
[完整數據](graded_static_results.json)；它是輸入場近似的界，不是解場誤差估計。
完整事件仍需檢查其影響。

FEM background h1 的 end 0.03125 mm 嘗試未通過原 1e-8 static relative residual
門檻，未接受其場，也未完成第二級。該次失敗沒有捕捉殘差的具體數值；
工具已補上失敗 snapshot／診斷保存。後續數值修正與背景控制見下一節。
BEM Q12／Q24 積分控制及 graded linear dynamic responses 亦是進入完整事件前的控制。
目前只證實上面的靜態差異與數值初始化改善，沒有同精度 FEM 速度或動態穩定性優勢。

![Endpoint controls; every norm includes the full contact interval](figures/graded_static_convergence.png)

```sh
PYTHONPATH=bem/src FDBEM_CORES=4 OPENBLAS_NUM_THREADS=1 JAX_ENABLE_X64=true \
  python -B bem/tools/bem_graded_static.py SOURCE_KERNEL \
  --restart-seed ORIGINAL_FLOAT64_SEED --outer-h 10 --contact-h 2 \
  --end-sizes .0625 .03125 .015625 .0078125 --static-backend uncondensed \
  --fem-studies FEM_H2_STATIC FEM_H2_FINE_STATIC --output FRESH_BEM_STATIC
```

此輪量測由 static driver 明確選用新 backend。後續新增
`simulate_refined_contact.py --static-initialization fem-body-active-set` 及
`check_initialized_contact_ad.py` 的對應接口；新 backend 的完整事件尚未執行完畢。

## FEM background h1 的殘差修正與完成結果

重跑捕捉到原 source 的相對殘差 1.02956e-8，確實略高於 1e-8 門檻。
以完全相同的 sparse matrix 修正原 split-force defect，第一次即降至 1.59e-10。
`fem_static_defect_correction.py` 現提供可重現的數值 adapter：先由原 solver 選擇
active set、共用其原 elastic assembly，然後在支承／platen 約束下修正
`K*u + C*u - rhs + spring`，要求每一步保留原 contact set，並達到更嚴的 1e-9 target。
它不修改原 source bindings、原 constitutive/contact kernels 或物理參數。
不能只看合併矩陣的 residual，因為 contact offset 與整體位移相減的 roundoff
在 split force 中不同；兩者的實際量測已保留。

三項數值測試核對 split-force 平衡、platen tie、拒絕 branch 改變和未收斂／寬鬆 target；
真實 h40 小模型亦保持原解。背景 h1 的三組真實控制現已完成：

|Endpoint size [mm]|FEM DOFs|Pressure peak [MPa]|Relative force residual|BEM/FEM pressure P1 L2|Shear P1 L2|
|---|---:|---:|---:|---:|---:|
|0.03125|442416|1437.4715|1.59e-10|1.6276%|0.6246%|
|0.015625|461222|1519.9783|2.54e-10|0.9162%|0.6230%|
|0.0078125|484790|1559.2922|5.31e-10|0.5849%|0.5084%|

最後兩級 FEM pressure／shear 的變化為 0.8539%／0.2187%。
最細兩組均只需一次 residual correction，所有組 active set 均未改變。
這些完整 static controls 共 113.67 s、峰值 process RSS 4699.83 MiB；
包含 mesh、JAX 組裝、原 static solve、數值修正與輸出，不是完整動態成本。
[原始結果、失敗紀錄及指紋](fem_background_static_results.json)包含 h2／h1 的全場比較。

BEM uncondensed static sensitivity 另以原 selected-set JAX root differentiation
求 `du/dp`，再用原 trace 與 `Kr*du/dp` 的量測 force projection 取得初始 trace／bulk-force
導數。小模型已與 condensed derivative、兩個 penalty 參數各自重新求解的 centered
finite differences 核對。transient／AD 接口保留這個量測到的初始 bulk-force 導數，
不用 condensed matrix 乘位移導數代替。包含 sliding static set、完整短程原始 Z-history
及重新初始化 finite differences 的四項測試通過；新 backend 的完整事件 AD 仍未完成。

最細 BEM end 0.0078125 mm 的相同網格 Q12／Q24 static control 亦完成：
pressure／shear 全接觸面 P1 L2 差為 4.30e-11／9.25e-12，
最大 absolute difference 為 2.02e-7／2.01e-8 MPa；Q24 full-body residual 為 1.97e-8 MPa。
[積分控制報告](graded_quadrature_results.json)保留原值、來源與指紋。
這只驗證 static quadrature，未驗證整個複數 CQ contour 的積分誤差。

候選 end 0.015625 mm 的 outer h10／h5 static control 亦完成。
pressure／shear 全接觸面 P1 L2 差為 5.54e-6／2.73e-5，通過原 1% 門檻；
見[完整結果](graded_outer_results.json)。這不是 dynamic outer-mesh convergence 證明。

## 連續時間的 linear dynamic response

`compare_linear_bulk_response.py` 現可明確套用相同 mesh adapter。
FEM h2／end 0.015625 與上述同尺寸端點 BEM 的三個頻率控制已完成。
`Re(s)=200 /s`，原 lumped FEM mass、支承、platen spring 與有限 penalty 均保留：

|Frequency|Pressure P1 L2|Shear P1 L2|Platen difference|
|---|---:|---:|---:|
|0 Hz|1.0708%|0.4428%|0.00250%|
|1000 Hz|1.1089%|0.4056%|0.01559%|
|10000 Hz|3.3997%|3.1473%|4.3129%|

三項場在三個取樣頻率都通過沿用的 5% linear screen；其原 FEM 方程 relative
residual 為 4.53e-11、1.34e-10、4.60e-12，BEM 為 4.00e-17、9.54e-16、1.89e-15。
[原始報告與指紋](graded_linear_results.json)保留所有值及輸入來源。
這是三個連續時間 closed-stick 線性控制，不含 CQ 誤差、branch switching、
slip weakening 或完整事件；也沒有證明高於 10 kHz 的所有頻率都通過。
