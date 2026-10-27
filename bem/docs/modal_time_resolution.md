# 時間積分的線性模態耗散診斷

另有[固定實際細網格的 bulk 時間符號診斷](bulk_time_symbol.md)，
直接比較兩有限 block 的連續時間及 BDF2 transfer，分離實際空間算子的時間誤差。
它仍是閉合黏著分支的頻率取樣，不能替代完整非線性事件的時間收斂。

2026-10-25。這是由既定時間離散式推導的解析診斷，**不是兩block的實測頻譜、
非線性誤差估計或13.2%滑移差的歸因證明**。頻率是測試值，沒有宣稱模型實際激發了它們。

對無阻尼模態`y'=i*omega*y`，令`z=i*omega*dt`。
BDF2主根與兩階段Radau IIA穩定函數為：

```text
(3-2*z)*xi^2 - 4*xi + 1 = 0
xi_BDF2 = (2+sqrt(1+2*z))/(3-2*z), xi_BDF2(0)=1
R_Radau = (1+z/3)/(1-2*z/3+z*z/6)
amplitude(T) = exp((T/dt)*log(abs(xi)))
```

Radau式可直接由本專案既定Butcher tableau的
`R(z)=1+z*b.T*inv(I-z*A)*ones(2)`得到。
BDF2取主離散模態，沒有把啟動時可能出現的寄生根當成物理模態。
下表的T=45 ms是假定同一模態在整個區間自由振動；真實rupture的激發時間與耗散不同。

|dt [µs]|BDF2，5 kHz振幅|BDF2，10 kHz振幅|Radau IIA，10 kHz振幅|
|---:|---:|---:|---:|
|4|0.509281|0.000061|0.538462|
|2|0.916850|0.259367|0.925158|
|1|0.989128|0.840613|0.990311|
|0.5|0.998632|0.978374|0.998783|

因此A-stability與小平衡殘差不能取代波動精度驗證。
接觸子步保持原bulk CQ網格，也不會改變這個線性bulk時間符號。
這支持先完成相同Q12的時間網格控制，再評估1 µs／更小dt或Radau的整體誤差與成本。
Radau的局部改善也不能直接當成完整接觸歷程三階收斂；
既有[短程Radau實驗](radau_experiment.md)已顯示積分不足的反例。

補讀Aldahham、Banjai，2025年出版、2022年預印本的
[Modified CQ與Galerkin BEM](https://arxiv.org/html/2203.00996v1)，重點為§3–6與§7.2–7.3。
它利用有限波速與延遲分解改善低階CQ，§7.2亦記錄trapezoidal修改式在積分不足時的不穩定。
本文驗證為純量聲學；本專案的P/S波、V/H/D耦合、凝縮與slip weakening須另外推導驗證。
因此目前沒有將modified CQ或trapezoidal切換成預設，也沒有借用其誤差／速度比。

完整解析數據含每步模、phase-rate差與公式，見
[modal_time_resolution_results.json](modal_time_resolution_results.json)。例如可用標準Python重算：

```python
import cmath
import math

dt, frequency, duration = 2e-6, 5000, 0.045
z = 2j * math.pi * frequency * dt
xi = (2 + cmath.sqrt(1 + 2 * z)) / (3 - 2 * z)
print(math.exp(duration / dt * math.log(abs(xi))))
```
