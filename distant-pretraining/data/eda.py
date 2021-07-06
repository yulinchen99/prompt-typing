# %%
import json
import matplotlib.pyplot as plt
with open('./distant_entity.json')as f:
    data = json.loads(f.read())
length = []
for e in data:
    length.append(len(data[e]))
# %%
plt.boxplot(length)
plt.show()
# %%
import numpy as np
np.max(length)
# %%
np.mean(length)
# %%
length2 = np.array(length)
plt.hist(np.log(length2), bins=20)
plt.show()
# %%
np.median(length)
# %%
