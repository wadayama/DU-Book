# Flux.jl 移行ガイド：v0.14以前 → v0.15以降

## 概要

Flux.jl v0.15以降で`Flux.Optimise`モジュールが非推奨となり、新しいOptimisers.jlベースのAPIに変更されました。
本ガイドはv0.16.4での動作を確認して作成されています。

## 基本的な変更

### 1. 勾配計算の変更

- 例１：`Chapter_2/Flux-demo.ipynb`

  **旧：**

  ```julia
  ps = Flux.params(x)
  gs = Flux.gradient(ps) do
      f(x)
  end
  x = x - 0.1 * gs[x]
  ```

  **新：**

  ```julia
  gs = Flux.gradient(f, x) # paramsの定義とdo-endが不要
  x = x - 0.1 * gs[1] # インデックスがxから1に変更
  ```

- 例２：`Chapter_2/NN-fullbatch.ipynb`

  **旧：**

  ```julia
  function GD(max_itr, η)
      W = randn(d, m)
      ret_list = zeros(max_itr)
      for i in 1:max_itr
          gs = Flux.gradient(Flux.params(W)) do
              L(W)
          end
          W = W - η*gs[W]
          ret_list[i] = L(W)
      end
      return ret_list
  end
  ```

  **新：**

  ```julia
  function GD(max_itr, η)
      W = randn(d, m)
      ret_list = zeros(max_itr)
      for i in 1:max_itr
          gs = Flux.gradient(L, W) # paramsの定義とdo-endが不要
          W = W - η*gs[1] # インデックスがWから1に変更
          ret_list[i] = L(W)
      end
      return ret_list
  end
  ```

### 2. オプティマイザーの変更

- 例１：`Chapter_2/Flux-demo.ipynb`

  **旧：**

  ```julia
  opt = Flux.Descent(0.1)
  x = x0
  for i in 1:10
      ps = Flux.params(x)
      gs = Flux.gradient(ps) do
          f(x)
      end
      Flux.Optimise.update!(opt, ps, gs) # ❌ Flux.Optimiseが非推奨、第一引数もoptからopt_stateに変更
      println(x)
  end
  ```

  **新：**

  ```julia
  opt = Flux.Descent(0.1)
  x = x0
  opt_state = Flux.setup(opt, x) # 事前のセットアップが必要
  for i in 1:10
      gs = Flux.gradient(f, x)
      Flux.update!(opt_state, x, gs[1]) # 勾配のインデックス指定が必要
      println(x)
  end
  ```

## 訓練ループの例

`Chapter_2/regression.ipynb`

**旧：**

```julia
p = [1.0,1.0, 1.0]
g(x,p) = p[1]*exp(-p[2]*x)+p[3]
opt = Flux.ADAM(0.4) 
loss(x,y) = (x-y)^2 
function train(data, p)
    ps = Flux.params(p)
    for d in data
        gs = Flux.gradient(ps) do
            y = d[2]
            y_hat = g(d[1],p)
            loss(y, y_hat)
        end
        Flux.Optimise.update!(opt, ps, gs) # ❌ Flux.Optimiseが非推奨、第一引数もoptからopt_stateに変更
    end
end
for i in 1:10
    train(DATA, p)
end
```

**新：**
```julia
p = [1.0,1.0, 1.0]
g(x,p) = p[1]*exp(-p[2]*x)+p[3]
opt = Flux.ADAM(0.4) 
loss(x,y) = (x-y)^2
function train(data, p)
    opt_state = Flux.setup(opt, p) # 事前のセットアップが必要
    for d in data
        gs = Flux.gradient((p)-> loss(d[2], g(d[1], p)), p)
        Flux.update!(opt_state, p, gs[1]) # 勾配のインデックス指定が必要
    end
end
for i in 1:10
    train(DATA, p)
end
```

## 複数パラメータの例

`Chapter_2/rbf.ipynb`

**旧：**

```julia
N = 100
T = 1.0
opt = Flux.ADAM(1.0) 
θ_train = ones(50)
train_itr = 250

function train(T, N, θ)
    ps = Flux.params(θ)
    for i in 1:train_itr
        gs = Flux.gradient(ps) do
            Fit(T, N, θ)
        end
        Flux.Optimise.update!(opt, ps, gs) # ❌ Flux.Optimiseが非推奨、第一引数もoptからopt_stateに変更
        println(Fit(T, N, θ))
    end
end
```

**新：**
```julia
N = 100
T = 1.0
opt = Flux.ADAM(1.0) 
θ_train = ones(50)
train_itr = 250

function train(T, N, θ)
    opt_state = Flux.setup(opt, θ) # 事前のセットアップが必要
    for i in 1:train_itr
        gs = Flux.gradient((T, N, θ)-> Fit(T, N, θ), T, N, θ)
        Flux.update!(opt_state, θ, gs[3]) # 勾配のインデックス指定が必要
        println(Fit(T, N, θ))
    end
end
```



## 重要なポイント

- `Flux.params()` → 直接パラメータを渡す
- `gs[param]` → `gs[1]`（数値インデックス）
- `Flux.Optimise.update!()` → `Flux.update!()`（事前セットアップが必要）
- 最初に`Flux.setup()`で状態を作成
- `Flux.update!()`は状態とパラメータを更新

## API変更の詳細

- v0.15以降、`Flux.Optimise.update!`は`Flux.update!`に変更
- 第一引数は`opt`（オプティマイザー）ではなく`opt_state`（セットアップ済み状態）

---
*作成：Lantian, Wei and Claude (Anthropic) - 2025年7月9日 v0.16.4での動作を確認して作成*

