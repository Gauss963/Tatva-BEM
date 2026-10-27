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
下一輪使用原 bulk h2 與 end sizes 1、0.5、0.25、0.125 mm，同時保留 unrefined baseline。
先看獨立連續加密差，再決定動態 reference，不能挑一個較接近 BEM 的 FEM 網格當成功。

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
