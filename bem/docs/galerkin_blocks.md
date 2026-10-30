# 不先建完整矩陣的 Galerkin block／fiber 取值

`tools/galerkin_blocks.py` 新增 exact selected-entry 介面，作為後續空間／頻率 ACA
的基礎。目前沒有加入低秩或頻率近似，也沒有替換正在執行的原始細網格 producer。

V 是 P0/P0，H=(M/2+K) 是 P0/P1，D 是 P1/P1。
相同整數 DOF 標籤在不同矩陣代表不同 basis roles；P1 節點還包含前一個面板的支撐。
介面因此先找出各場需要的 supporting panel pairs，再只準備那些原 quadrature batches。
不先建立 O(n²) pair geometry、完整 mixed mass 或完整 V/H/D。
請求本身若涵蓋完整輸出，則仍會產生完整資料；statistics 會明確記錄。

原 `PreparedGalerkinOperators.assemble` 的每個局部數值 AST 直接沿用，
包括 Green/Bessel、traction、self/adjacent Duffy、log subtraction 與解析 moment 修正。
介面只移除完整矩陣配置與原 scatter，另回傳局部量供選定區塊 scatter。
原 allocation、reciprocal scatter、M/2 suffix 的 AST 皆有 guard；未識別的變動即拒絕。
Panel pairs 保留原 self／adjacent／disjoint 順序和原 batch boundaries，
V/D 的 reciprocal scatter 使用 bilinear transpose，沒有 complex conjugation。
原 module bindings 不改動。選定的幾何與 maps 為唯讀，可跨頻率與執行緒共用。

## 驗證範圍

19 項本機測試通過，包括全部 pair-group ordinals、小／大／純實頻率、兩個 body、
self／wrap-corner／far blocks、整欄 fiber、unsorted indices、分塊重組完整矩陣、
complex bilinear reciprocity、四個並行唯讀呼叫，以及兩物體凝縮／完整邊界重建。
測試也禁止 selected-entry 路徑呼叫完整 prepared constructor 或配置完整矩陣，
並驗證 pair budget 在幾何配置之前生效。

2026-10-30 的實際細網格 moving block 使用 outer 10 mm、contact 2 mm、
end/corner 0.015625 mm、Q12，共 490 panels／980 DOFs。
原完整組裝包含 120295 panel pairs，每個矩陣 960400 entries。
一次選擇兩個 4×4 blocks（corner／separated），加上一個 980×1 column fiber：

|請求|準備的 panel pairs|quadrature points|準備幾何 array bytes|
|---|---:|---:|---:|
|corner 4×4|6|1152|76368|
|separated 4×4|12|1728|115872|
|完整 column fiber|979|141408|9480872|

這三個請求的幾何準備合計 0.172 s；原完整幾何準備 12.192 s。
下面的時間是「只需要上述選定 entries」時，直接取值與先組完整矩陣再取值的比較。
**輸出數量不同於完整矩陣；不能把此比值當作 ACA、CQ 或完整事件的加速比。**

|2 µs／45 ms Radau 計畫中的實際 s（1/s）|完整單 body 組裝中位數|三個選定請求合計中位數|
|---|---:|---:|
|175.6733 + 0i|28.766 s|0.313 s|
|175.7813 + 31398.6686i|56.105 s|0.559 s|
|2999473.0418 + 0i|27.718 s|0.284 s|

每個頻率做兩次交替順序，BLAS 單執行緒；原 BEM 暫停後才開始，完成後恢復。
三個請求的 V/H/D/M 全部與獨立完整組裝逐項比較：兩個實頻全部 bitwise 相同，
複頻最大絕對差 1.39e-17、最大 selected-block relative Frobenius 差 8.64e-21。
這些是被請求 entries 的誤差，沒有拿它代替未取樣 entries 或壓縮後解的誤差。
全組 benchmark 連準備／重複／檢查 239.92 s。

[完整結果與來源指紋](galerkin_blocks_results.json)保留每次檢查及成本。
第一次測試因 reference mixed-mass 欄位名稱寫成 `mass` 而失敗；原紀錄保留，
修正 public block 名稱為 `mixed_mass` 後重新通過。該失敗成本未從 wall cost 隱藏。

```python
# Requires PYTHONPATH=bem/src; experimental tools are loaded explicitly.
import runpy
import numpy as np

api = runpy.run_path("bem/tools/galerkin_blocks.py")
provider = api["GalerkinBlockProvider"](mesh, material, quadrature=12)
fiber = provider.request(np.arange(mesh.dofs), [1], max_pairs=2000)
values = fiber.evaluate(200 + 20000j)
# values.single_layer, .trace_double_layer, .hypersingular, .mixed_mass
```

下一步須區分 P0／P1 的 spatial trees，測量各頻率 far blocks 的可壓縮 rank，
並檢查壓縮誤差經 V solve、凝縮、CQ、非線性接觸與 initialized AD 後的傳播。
有限體共振、非單調摩擦與近場／端點仍保留原精度門檻。
目前還沒有完整 ACA kernel、低秩轉置求解或同精度 FEM 優勢的證據。
