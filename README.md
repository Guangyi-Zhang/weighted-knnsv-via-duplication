# Project

## Examples

More usage can be found in `shapley` function in `knnsvdup/dup.py`.

```python
>>> import numpy as np
>>> from functools import partial
>>> from knnsvdup.helper import kernel_value
>>> from knnsvdup.dup import shapley
>>> D = [
...     (np.array([0.5]), 1),
...     (np.array([2.0]), 1),
...     (np.array([1.0]), 0)
... ]
>>> Z_test = [(np.array([0.0]), 1),]
>>> kernel_fn = partial(kernel_value, sigma=1)
>>> values = shapley(D, Z_test, K=1, value_type="dup", kernel_fn=kernel_fn)
>>> print(values)
[ 3.3873862   1.39327479 -4.28066099]

```

## Tests

```bash
pytest -s -vv --doctest-glob="*.md"
```
