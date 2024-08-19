# Modified DH Parameter in YAIK

The modified DH parameter is shown in `docs/modified_dh.png`, but the DH parameter is not indexed in the same way. The DH parameter in the code is

```python
class DHEntry:  
	alpha # alpha_{i-1} in the figure
  a # a_{i-1} in the figure
  theta # theta_i in the figure
  d # d_i in the figure
```

This `DHEntry` is **modified** DH parameters. When using as modified DH parameter, the transformation is
$$
T_{i}^{i-1} = \text{rotx}(\alpha_{i-1})~\text{trans\_x}(a_{i-1})~\text{rotz}(\theta_i)~\text{trans\_z}(d_i)
$$
where $T_{i}^{i-1}$ maps coordinate vectors in $i$-th frame (child frame) to the $(i-1)$-th frame parent frame. In python code, it is written as

```python
entry_i = DHEntry()
T_{i}_to_{i-1} = rotx(entry_i.alpha) * trans_x(entry_i.a) * rotz(entry_i.theta) * trans_z(entry_i.d)
```

