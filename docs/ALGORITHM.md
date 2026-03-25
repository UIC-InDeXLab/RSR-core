# RSR Algorithm: Redundant Segment Reduction

## Problem Statement

Compute the binary matrix-vector product `y = M · v`, where:
- `M` is an `n × n` binary matrix (entries are 0 or 1)
- `v` is a real-valued vector of length `n`
- `y` is a real-valued output vector of length `n`

The naive approach costs O(n²) multiply-adds. RSR exploits the fact that binary matrices (especially in quantized neural networks) often have many **duplicate columns**. Columns that are identical contribute identically to each output row, so their input vector elements can be **summed first** and multiplied only once — reducing total work proportional to the number of unique column patterns.

---

## Key Idea

For a binary matrix, a set of columns with the **same bit pattern** contributes to the output identically: each of those columns adds the same set of rows. If columns `j1, j2, j3` all have the pattern `[1, 0, 1]ᵀ`, then their joint contribution to the output is:

```
y[0] += v[j1] + v[j2] + v[j3]
y[2] += v[j1] + v[j2] + v[j3]
```

Instead of three separate additions per output row, RSR computes the scalar sum `s = v[j1] + v[j2] + v[j3]` once and then distributes it to rows 0 and 2. This is the **Aggregate then Scatter** pattern.

---

## Block Decomposition

Processing all `n` rows at once would require encoding each column as an `n`-bit integer — impractical for large `n`. RSR instead splits `M` horizontally into blocks of `k` rows each.

**Parameters:**
- `n`: matrix dimension (must be square)
- `k`: block height (number of rows per block); `n` must be divisible by `k`
- `num_blocks = n / k`

Each block is processed independently. Block `b` covers rows `[b·k, (b+1)·k)` of `M`. Columns are encoded as `k`-bit integers within each block, so the maximum number of unique patterns per block is `2^k`.

The outputs from all blocks are concatenated to produce the final `n`-dimensional result.

---

## Algorithm: Two Phases

### Phase 1 — Preprocessing (one-time, per matrix)

Done once when the matrix `M` is loaded. The result is a set of data structures that accelerate all future inferences.

**For each block `b` (rows `b·k` to `(b+1)·k - 1`):**

#### Step 1.1 — Encode columns as integers

Extract the sub-matrix `block = M[b·k : (b+1)·k, :]` of shape `(k, n)`.

Encode each column as a `k`-bit integer where **row 0 is the most significant bit (MSB)**:

```
col_value[j] = sum over row i of: M[b·k + i, j] * 2^(k-1-i)
```

Example with `k=3`: column `[1, 0, 1]ᵀ` → `1·4 + 0·2 + 1·1 = 5`.

Using dot product notation:
```
bit_weights = [2^(k-1), 2^(k-2), ..., 2^1, 2^0]   # shape (k,)
col_values = bit_weights @ block                      # shape (n,)
```

#### Step 1.2 — Sort columns by their encoded value

Compute a permutation `perm` that sorts columns by `col_values`:

```
perm = argsort(col_values, stable=True)   # shape (n,)
```

After applying `perm`, columns with the same bit pattern are grouped together.

#### Step 1.3 — Find unique patterns and group boundaries

Apply `perm` to `col_values` to get `sorted_values`, then find unique values:

```
sorted_values = col_values[perm]
uniq, inverse = unique(sorted_values)
```

- `uniq`: array of unique k-bit integer patterns, shape `(num_unique,)`
- `inverse`: for each position in `sorted_values`, its index into `uniq`, shape `(n,)`

The `inverse` array (called `group_indices`) maps each sorted column position to its pattern group.

Group boundaries: since columns are sorted, all columns with pattern group `g` occupy a contiguous range in `perm`. Compute cumulative counts to find these boundaries:

```
counts[g] = number of columns with group index g
group_ends[g] = sum of counts[0..g]   # cumulative, i.e. group g covers [group_ends[g-1], group_ends[g])
```

#### Step 1.4 — Decode unique patterns to output row indices

For each unique pattern integer `pat`, find which bit positions are set — these are the output row indices to update during scatter:

```
for each unique pattern pat:
    scatter_rows_for_pat = [k-1-bit for each set bit in pat]
```

The mapping from bit position to row index: bit position `b` (0 = LSB) in the integer corresponds to row index `k-1-b` in the block (because row 0 is MSB).

Store a flat array `scatter_rows` and an offset array `scatter_offsets` so that pattern `g`'s output rows are `scatter_rows[scatter_offsets[g] : scatter_offsets[g+1]]`.

#### Preprocessing outputs (per block `b`):

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `perms[b]` | `(n,)` | int | Column permutation sorting by pattern |
| `group_ends[b]` | `(num_unique_b,)` | int | Cumulative group end indices into `perm` |
| `scatter_offsets[b]` | `(num_unique_b + 1,)` | int | Offsets into `scatter_rows` per group |
| `scatter_rows[b]` | `(total_set_bits_b,)` | int | Output row indices (within block) for each group |

---

### Phase 2 — Inference (per input vector `v`)

Given preprocessed data and input vector `v` of shape `(n,)`:

#### Step 2.1 — Permute the input vector

For each block `b`, reorder `v` according to `perms[b]`:

```
v_perm[b] = v[perms[b]]   # shape (n,)
```

After permutation, elements that belong to the same column group are contiguous.

#### Step 2.2 — Aggregate (sum within each group)

For each block `b` and each group `g`:

```
start = group_ends[g-1]  (or 0 if g == 0)
end   = group_ends[g]
aggregated[g] = sum(v_perm[b][start : end])
```

This sums all input vector elements corresponding to columns with the same pattern.

#### Step 2.3 — Scatter (distribute to output rows)

For each block `b`, initialize output slice `out[b·k : (b+1)·k] = 0`.

For each group `g`:
```
s_begin = scatter_offsets[g]
s_end   = scatter_offsets[g + 1]
for s in range(s_begin, s_end):
    out[b·k + scatter_rows[s]] += aggregated[g]
```

This distributes the aggregated sum to every output row where the group's pattern has a 1-bit.

#### Step 2.4 — Return output

Concatenate results from all blocks: `y = out[0:n]`.

---

## Pseudocode

### Preprocessing

```python
def preprocess(M, k):
    n = M.shape[0]
    assert n % k == 0
    num_blocks = n // k

    perms = []
    group_ends_list = []
    scatter_offsets_list = []
    scatter_rows_list = []

    for b in range(num_blocks):
        block = M[b*k : (b+1)*k, :]          # shape (k, n)

        # Step 1.1: encode columns
        bit_weights = [2**(k-1-i) for i in range(k)]
        col_values = [sum(bit_weights[i] * block[i, j] for i in range(k))
                      for j in range(n)]      # length n

        # Step 1.2: sort columns
        perm = argsort(col_values, stable=True)

        # Step 1.3: find groups
        sorted_values = [col_values[perm[j]] for j in range(n)]
        uniq = sorted(set(sorted_values))
        pat_to_idx = {pat: idx for idx, pat in enumerate(uniq)}
        num_unique = len(uniq)

        counts = [0] * num_unique
        for j in range(n):
            counts[pat_to_idx[sorted_values[j]]] += 1

        group_ends = []
        cum = 0
        for g in range(num_unique):
            cum += counts[g]
            group_ends.append(cum)

        # Step 1.4: decode patterns
        scatter_offsets = [0]
        scatter_rows = []
        for pat in uniq:
            rows = []
            for bit_pos in range(k):          # bit_pos 0 = LSB of integer
                if pat & (1 << bit_pos):
                    rows.append(k - 1 - bit_pos)  # row 0 = MSB = highest bit_pos
            scatter_rows.extend(rows)
            scatter_offsets.append(len(scatter_rows))

        perms.append(perm)
        group_ends_list.append(group_ends)
        scatter_offsets_list.append(scatter_offsets)
        scatter_rows_list.append(scatter_rows)

    return perms, group_ends_list, scatter_offsets_list, scatter_rows_list
```

### Inference

```python
def infer(v, perms, group_ends_list, scatter_offsets_list, scatter_rows_list, n, k):
    num_blocks = n // k
    out = [0.0] * n

    for b in range(num_blocks):
        perm = perms[b]
        group_ends = group_ends_list[b]
        scatter_offsets = scatter_offsets_list[b]
        scatter_rows = scatter_rows_list[b]
        num_unique = len(group_ends)

        # Step 2.1: permute input
        v_perm = [v[perm[j]] for j in range(n)]

        # Step 2.2: aggregate
        aggregated = []
        start = 0
        for g in range(num_unique):
            end = group_ends[g]
            agg = sum(v_perm[start:end])
            aggregated.append(agg)
            start = end

        # Step 2.3: scatter
        for g in range(num_unique):
            s_begin = scatter_offsets[g]
            s_end = scatter_offsets[g + 1]
            for s in range(s_begin, s_end):
                out[b*k + scatter_rows[s]] += aggregated[g]

    return out
```

---

## Concrete Example

`n=4`, `k=2`, matrix:

```
M = [[1, 0, 1, 0],
     [0, 1, 1, 0],
     [1, 1, 0, 0],
     [0, 0, 1, 1]]
```

`num_blocks = 2`. Input vector `v = [1, 2, 3, 4]`.

**Block 0** (rows 0–1):
```
block = [[1, 0, 1, 0],
         [0, 1, 1, 0]]

bit_weights = [2, 1]
col_values = [2*1+1*0, 2*0+1*1, 2*1+1*1, 2*0+1*0] = [2, 1, 3, 0]

perm = argsort([2,1,3,0]) = [3, 1, 0, 2]   (indices sorted by value)
sorted_values = [0, 1, 2, 3]   (all unique)
uniq = [0, 1, 2, 3]
group_ends = [1, 2, 3, 4]

Pattern 0 (binary 00): no bits set → scatter_rows = []
Pattern 1 (binary 01): bit 0 set → row k-1-0 = 1 → scatter_rows = [1]
Pattern 2 (binary 10): bit 1 set → row k-1-1 = 0 → scatter_rows = [0]
Pattern 3 (binary 11): bits 0,1 set → rows 1, 0 → scatter_rows = [1, 0]

scatter_offsets = [0, 0, 1, 2, 4]
scatter_rows    = [1, 0, 1, 0]

Inference:
  v_perm = v[[3,1,0,2]] = [4, 2, 1, 3]
  aggregated = [v_perm[0:1], v_perm[1:2], v_perm[2:3], v_perm[3:4]]
             = [4, 2, 1, 3]
  Scatter:
    group 0 (agg=4): no rows → nothing
    group 1 (agg=2): row 1 → out[1] += 2
    group 2 (agg=1): row 0 → out[0] += 1
    group 3 (agg=3): rows 1,0 → out[1] += 3, out[0] += 3
  out[0:2] = [4, 5]
```

**Block 1** (rows 2–3): analogous → `out[2:4]` computed similarly.

**Verification**: `out[0] = M[0,:]·v = 1·1+0·2+1·3+0·4 = 4`. ✓

---

## Mathematical Correctness

For block `b`, the standard computation of output row `r` is:

```
out[b·k + r] = sum over j of M[b·k + r, j] * v[j]
```

RSR groups columns by their pattern. For group `g` (pattern `pat`), let `C_g` be the set of column indices in that group. All columns in `C_g` satisfy `M[b·k + r, j] = pat[r]` for all `j ∈ C_g`. Therefore:

```
sum_{j in C_g} M[b·k + r, j] * v[j]
  = pat[r] * sum_{j in C_g} v[j]
  = pat[r] * aggregated[g]
```

Summing over all groups reproduces the full dot product.

---

## Extension to Non-Square Matrices

### Problem Statement

The algorithm above assumes `M` is square (`n × n`). In practice, weight matrices in neural networks are typically non-square: `M` is `n_rows × n_cols`, `v` has length `n_cols`, and the output `y` has length `n_rows`.

### Changes

The generalization is straightforward — only the block dimensions change:

1. **Block decomposition**: `M` is split into horizontal blocks of `k` rows each. Block `b` is `M[b·k : (b+1)·k, :]` with shape `(k, n_cols)` instead of `(k, n)`. The number of blocks is `num_blocks = n_rows / k`.

2. **Column encoding**: Each of the `n_cols` columns is encoded as a `k`-bit integer (unchanged logic, just iterating over `n_cols` columns instead of `n`).

3. **Permutation arrays**: Each block's permutation `perms[b]` has `n_cols` entries (one per column), not `n`.

4. **Input vector**: `v` has length `n_cols`. The permuted vector `v_perm[b]` also has `n_cols` entries.

5. **Output vector**: `y` has length `n_rows = num_blocks · k`. The scatter phase writes to rows within each `k`-sized block, concatenated to form the full output.

6. **Row padding**: If `n_rows` is not divisible by `k`, pad `M` with zero rows to the nearest multiple of `k` and trim the output back to `n_rows` after inference.

Everything else — the aggregate-then-scatter pattern, the counting sort, the group structure — remains identical. The inference kernel does not need to know `n_rows` at all; it only needs `n_cols` (as the permutation stride), `k`, and `num_blocks`.

---

## Extension to Ternary Matrices (1.58-bit)

### Problem Statement

Compute `y = M · v` where `M` is an `n × n` **ternary** matrix with entries in `{-1, 0, +1}`.

### Reduction to Two Binary RSRs

A ternary matrix `M` can be decomposed into two binary matrices:

```
M_pos[i, j] = 1  if M[i, j] = +1,  else 0
M_neg[i, j] = 1  if M[i, j] = -1,  else 0
```

These satisfy `M = M_pos - M_neg`, so:

```
y = M · v = M_pos · v − M_neg · v
```

Each of `M_pos` and `M_neg` is a binary `{0, 1}` matrix, so the standard binary RSR algorithm applies to each independently.

### Unified Ternary Encoding

Rather than maintaining two separate RSR data structures, the ternary extension uses a **single 2k-bit column encoding** per block:

For block `b` (rows `b·k` to `(b+1)·k - 1`), each column `j` is encoded as a `2k`-bit integer:

```
ternary_code[j] = (pos_pattern << k) | neg_pattern
```

where:
- `pos_pattern`: k-bit integer encoding which rows have `M[i, j] = +1` (same MSB convention as binary RSR)
- `neg_pattern`: k-bit integer encoding which rows have `M[i, j] = -1`

This encoding has at most `3^k` unique values per block (since each row position can be `+1`, `-1`, or `0`). Columns are sorted by their ternary code, grouped by unique code, and then:

1. **Aggregate**: sum `v[j]` for all columns `j` in the same group (identical to binary RSR)
2. **Signed Scatter**: for each set bit in `pos_pattern`, add the aggregate to the output row; for each set bit in `neg_pattern`, subtract it

### Preprocessing (per block `b`)

```python
M_pos = (M == +1)    # binary
M_neg = (M == -1)    # binary

bit_weights = [2^(k-1), ..., 2^0]
pos_pattern = bit_weights @ M_pos[b*k:(b+1)*k, :]
neg_pattern = bit_weights @ M_neg[b*k:(b+1)*k, :]
ternary_code = (pos_pattern << k) | neg_pattern

perm = argsort(ternary_code)   # or counting sort
# group_ends, scatter_rows, scatter_signs computed from unique codes
```

### Inference (per input vector `v`)

```python
for each block b:
    v_perm = v[perm]                              # permute input
    for each group g:
        agg = sum(v_perm[group_start:group_end])  # aggregate
        for each (row, sign) in scatter_entries[g]:
            out[b*k + row] += sign * agg          # signed scatter
```

### Preprocessing outputs (per block `b`)

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `perms[b]` | `(n,)` | int | Column permutation sorting by ternary code |
| `group_ends[b]` | `(num_unique_b,)` | int | Cumulative group end indices |
| `scatter_offsets[b]` | `(num_unique_b + 1,)` | int | Offsets into scatter arrays per group |
| `scatter_rows[b]` | `(total_entries_b,)` | int | Output row indices (within block) |
| `scatter_signs[b]` | `(total_entries_b,)` | int | `+1` or `-1` per scatter entry |

### Trade-offs vs. Two Separate Binary RSRs

| Approach | Permutations | Aggregations | Scatter passes | Max unique patterns |
|----------|-------------|-------------|----------------|-------------------|
| Two binary RSRs | 2 per block | 2 per block | 2 per block | 2^k each |
| Unified ternary | 1 per block | 1 per block | 1 per block | 3^k |

The unified approach halves the number of permutation and aggregation passes at the cost of a larger pattern space. For typical `k` values (4–8), `3^k` remains manageable (81–6561 patterns).

### Non-Square Ternary Matrices

The non-square generalization described in the "Extension to Non-Square Matrices" section applies identically to the ternary case. For a ternary matrix `M` of shape `(n_rows, n_cols)`, each block has `n_cols` columns encoded as `2k`-bit integers, permutations have `n_cols` entries per block, and the output has `n_rows` elements. Row padding to a multiple of `k` is applied if needed.