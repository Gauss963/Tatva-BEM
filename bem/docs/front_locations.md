# 破裂起始位置與輸出解析度

2026-10-25。比較器分別保存原生BEM節點與插值到FEM節點的first-passage表格。
先插值累積滑移，再找門檻通過時間，可能漏掉兩個共同節點之間最早滑動的BEM節點。
因此不能將既有共同網格曲線的最早位置，直接當成原生BEM的成核位置。

原FEM存檔的最早micro、0.1Dc及Dc通過位置均為y=0mm。
既有2us/Q12一子步的共同網格曲線分別給出y=2、20及12mm；
這是值得檢查空間誤差的線索，仍需新增的原生節點結果才能區分插值影響。
微滑移門檻為1e-7mm；Dc=0.0125959mm。這些量不是瞬時破裂速度。

`compare_fronts.py`增加每組`.native.csv`，原共同網格CSV格式保留。
JSON分開保存`bem_native_node_earliest`、`bem_common_grid_earliest`與
`fem_stored_node_earliest`。若多個節點在同一保存frame首次通過門檻，
會保留全部並列位置及前後frame時間區間；線性插值的最小時間另列為估計，
不將它當成已解析的唯一先後順序。非有限輸入會報錯，避免誤記為尚未到達。

從repository根目錄執行，輸出使用新資料夾：

```sh
FDBEM_CORES=4 bem/run.sh front-compare RUN_A RUN_B \
  --reference-table validation/static_bem2d/reference/fem_front_arrivals.csv \
  --output bem/outputs/new-front-comparison
```

上述命令需plot選配依賴；參考表可離線使用，不必再次讀取原始FEM HDF5。
比較累積滑移幅度、牽引分布與到達時間時，仍須同步檢查時間步、接觸子步、
網格、初始seed及保存間距，不能只用一個最早位置判定收斂。
