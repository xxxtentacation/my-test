# F2 || $\sum W_j C_j$ 动态规划算法设计

基于最优解的分块结构性质，参考 F2|r_j|Cmax PTAS（Hall 1998 / Kovalyov & Werner 1997）中**区间划分 + 状态压缩**的动态规划思想，设计如下算法。算法按四个阶段展开，每阶段前列出所用符号。

---

## 问题性质（来自 note.md）

1. **同排序性质**：两台机器上工件加工顺序一致（Permutation Schedule），已有经典结论。
2. **权重分块**：将工件按权重分为 $K$ 类，$W_1 > W_2 > \dots > W_K$。最优调度可划分为 $K$ 个连续逻辑区间（块）。
3. **位置约束**：$\pi(J_k) \le k$，即第 $k$ 类工件只能出现在前 $k$ 个块中。
4. **块尾约束**：第 $k$ 块的最后一个工件权重为 $W_k$。
5. **块内排序**：每个块内部的工件遵循 SPT(1)-LPT(2) 规则（Johnson 序）。

---

## 与 F2|r_j|Cmax PTAS 的对应关系

| F2\|r_j\|Cmax PTAS | 本算法 |
|---|---|
| 按释放日期 $r_j$ 划分区间 | 按权重 $W_k$ 划分区间（块） |
| 工件只能放入 $r_j$ 对应的区间及之后 | 工件只能放入权重类对应的块及之前 ($\pi(J_k) \le k$) |
| 区间内按 Johnson 序排列 | 块内按 Johnson 序排列 |
| DP 状态记录各区间的 M2 完工时间 | DP 状态记录各块的 A（M1 总时长）与 B（M2 完工时间） |
| 目标：最小化 $C_{\max}$ | 目标：最小化 $\sum W_j C_j$（需累加各工件完工时间） |

---

## 阶段一：预处理

**本阶段符号：**

| 符号 | 含义 |
|---|---|
| $n$ | 工件总数 |
| $K$ | 权重类别数 |
| $J_k$ | 权重为 $W_k$ 的工件集合（$k = 1,\dots,K$） |
| $W_k$ | $J_k$ 的公共权重，$W_1 > W_2 > \dots > W_K$ |
| $n_k$ | $\vert J_k \vert$，第 $k$ 类工件数 |
| $a_j$ | 工件 $j$ 在机器 1 上的加工时间 |
| $b_j$ | 工件 $j$ 在机器 2 上的加工时间 |
| $w_j$ | 工件 $j$ 的权重（$j \in J_k \iff w_j = W_k$） |
| $L_k$ | $J_k$ 按 Johnson 序排列后的有序列表 |

**操作：**

```
1. 将工件按权重分组，得到 J_1, J_2, ..., J_K (W_1 > W_2 > ... > W_K)
2. 对每组 J_k 内部按 SPT(1)-LPT(2) 排序，得到有序列表 L_k = [j_{k,1}, ..., j_{k,n_k}]
   （比较规则：x 在 y 之前 ⇔ min(a_x, b_y) < min(a_y, b_x)，或相等时 a_x ≤ a_y）
```

Johnson 排序（Merge\_By\_Johnson）的实现：

```
Merge_By_Johnson(T):
    T1 = {j ∈ T | a_j ≤ b_j}，按 a_j 非降序排列
    T2 = {j ∈ T | a_j > b_j}，按 b_j 非增序排列
    return T1 拼接 T2
```

---

## 阶段二：状态定义

**本阶段符号：**

| 符号 | 含义 |
|---|---|
| $\mathcal{Y}^{(k)}$ | 处理完前 $k$ 个块后的非支配状态集合（$k = 0,\dots,K$） |
| $P$ | 前缀向量 $(p_{k+1},\dots,p_K)$，$p_m =$ 已从 $L_m$ 取用的工件数 |
| $A$ | 前 $k$ 个块在 M1 上的累计加工时间 |
| $B$ | 第 $k$ 个块在 M2 上的完工时间（$k{+}1$ 块的 M2 可用时刻） |
| $z$ | 前 $k$ 个块的累计加权完工时间和 $\sum w_j C_j$ |

**状态定义：**

处理完前 $k$ 个块后，状态元组为：

$$(P,\; A,\; B,\; z)$$

其中 $P = (p_{k+1}, p_{k+2}, \dots, p_K)$，$0 \le p_m \le n_m$。由位置约束 $\pi(J_k) \le k$，处理完第 $k$ 块后 $J_1 \dots J_k$ 必已全部取用完毕（即 $p_m = n_m$ 对所有 $m \le k$ 成立），因此 $P$ 只需记录 $m > k$ 的类。

**初始状态：**

$$\mathcal{Y}^{(0)} = \{ (\mathbf{0},\; 0,\; 0,\; 0) \}$$

---

## 阶段三：块计算（Compute\_Block）

**本阶段符号：**

| 符号 | 含义 |
|---|---|
| $D$ | 选择向量 $(d_{k+1},\dots,d_K)$：后续（k+1到K）每个状态（权重类别）中剩余的工件数的集合， |
| $d_{k+1}$ | $= n_{k+1} - p_{k+1}$，第 $k{+}1$ 类剩余工件数（强制全部放入本块） |
| $T$ | 放入当前块的工件集合 |
| $\sigma$ | $T$ 按 Johnson 序排列后的序列 |
| $A', B'$ | 块内遍历更新时的 M1 累计时间与 M2 运行完工时间（最终作为块状态输出） |
| $P'$ | 块结束后的前缀向量（输出） |
| $z'$ | 块内遍历更新时的累计成本（最终作为块状态输出） |

**操作：**

给定状态 $(P, A, B, z) \in \mathcal{Y}^{(k)}$，为第 $k{+}1$ 块选择工件：从每个 $L_m$（$m \ge k{+}1$）取前缀 $L_m[p_m{+}1 \dots p_m{+}d_m]$，其中 $d_{k+1} = n_{k+1} - p_{k+1}$（强制全部取完），$d_m \ge 0$（$m > k{+}1$）。

```
Compute_Block(k, P, A, B, z, D):
    // 1. 收集工件
    T = ∅
    for m = k+1 to K:
        for i = 1 to d_m:
            T = T ∪ { L_m[p_m + i] }
    
    // 2. Johnson 排序
    σ = Merge_By_Johnson(T)
    
    // 3. 验证块尾约束：序列末工件必须来自 J_{k+1}
    if weight(σ[|σ|]) ≠ W_{k+1}: return INVALID
    
    // 4. 状态滚动更新
    A' = A, B' = B, z' = z
    for each job j in σ (按序):
        A' = A' + a_j
        B' = max(B', A') + b_j  // B' 就是当前工件的 M2 完工时间
        z' = z' + w_j · B'
    
    // 5. 返回更新值
    return (A', B', z')
```

---

## 阶段四：状态转移与完整算法

**本阶段符号：**

| 符号 | 含义 |
|---|---|
| $A_{\max}$ | $\sum_{j=1}^n a_j$，M1 总时长的上界 |
| $B_{\max}$ | $\sum_{j=1}^n (a_j + b_j)$，M2 完工时间的上界 |
| $Z^*$ | 最优（最小）总加权完工时间 |

其余符号 $P, A, B, z, D, A', B', P', z'$ 继承前文含义。

**状态转移：**

对每个 $k = 0 \to K-1$，从 $\mathcal{Y}^{(k)}$ 转移到 $\mathcal{Y}^{(k+1)}$：

```
for each (P, A, B, z) ∈ Y^(k):
    d_{k+1} = n_{k+1} - p_{k+1}                   // 第 k+1 类剩余工件（强制）
    for each D = (d_{k+1}, d_{k+2}, ..., d_K) where 0 ≤ d_m ≤ n_m - p_m:
        result = Compute_Block(k, P, A, B, z, D)
        if result == INVALID: continue
        
        (A', B', z') = result
        P' = (p_{k+2}+d_{k+2}, ..., p_K+d_K)   // 注意：p_{k+1} 维度已满，消去
        
        // 占优剪枝：若 Y^(k+1) 中已有相同 (P', A', B') 且 z_hat ≤ z'，则丢弃
        Insert (P', A', B', z') into Y^(k+1), pruning dominated states
```

**提取最优：**

$$Z^* = \min \{\, z \mid (\emptyset, A, B, z) \in \mathcal{Y}^{(K)} \,\}$$

---

## 完整伪代码

```text
Algorithm DP_F2_WeightedSum(J, A, B, W):

    // ===== 阶段一：预处理 =====
    Group jobs by weight → J_1..J_K (W_1 > ... > W_K)
    For k = 1 to K:
        L_k = Merge_By_Johnson(J_k)

    // ===== 阶段二：初始化 =====
    Y^(0) = { (0, 0, 0, 0) }

    // ===== 阶段三/四：逐块 DP =====
    for k = 0 to K-1:
        Y^(k+1) = ∅
        for each (P, A, B, z) ∈ Y^(k):
            d_{k+1} = n_{k+1} - p_{k+1}
            for each D = (d_{k+1}, d_{k+2}, ..., d_K) in valid range:
                result = Compute_Block(k, P, A, B, z, D)
                if result == INVALID: continue
                
                (A', B', z') = result
                P' = (p_{k+2}+d_{k+2}, ..., p_K+d_K)
                
                if no (P', A', B', z_hat) ∈ Y^(k+1) with z_hat ≤ z':
                    Insert and prune Y^(k+1)

    // ===== 提取最优 =====
    return min{ z | (∅, A, B, z) ∈ Y^(K) }
```

---

## 复杂度分析

- **状态空间**：$P$ 的组合数 $\prod_{m=k+1}^K (n_m+1) = O(n^{K-k})$；$A = O(A_{\max})$；$B = O(B_{\max})$。
  每块状态数：$|\mathcal{Y}^{(k)}| = O(n^{K-k} \cdot A_{\max} \cdot B_{\max})$。

- **转移开销**：枚举 $D$ 共 $O(n^{K-k-1})$ 种；Compute\_Block 中 Johnson 排序 $O(n \log n)$。

- **总复杂度**：
  $$O\left(n^{2K-1} \cdot A_{\max} \cdot B_{\max} \cdot \log n\right)$$

  此为**伪多项式时间**。当权重种类 $K$ 固定时，复杂度为 $n, A_{\max}, B_{\max}$ 的多项式。

- **PTAS 路径**：对权重做几何舍入使 $K = O(\frac{1}{\varepsilon}\log W_{\max})$，对加工时间做倍数舍入使 $A_{\max}, B_{\max} = O(n^2/\varepsilon)$，即可在 $\text{poly}(n, 1/\varepsilon)$ 时间内得到 $(1+\varepsilon)$ 近似解。
