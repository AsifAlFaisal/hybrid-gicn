#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
# %%
df = pd.read_csv("../Data/merged_prop.csv")
props = ['LogP','MP','BP','WS','VP','HL','AOH','KOC','BCF','KM','KOA','BioHL']
prop = props[11]
df = df.loc[df[prop].notnull(), ['SMILES','InChI', prop]]
df.head(2)

#%%
X = df[['SMILES','InChI']]
y = df[prop]

# rs = 31 for BCF, KM, KOA | rs=32 for VP | 30 for others
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=30) 
# %%
train = pd.DataFrame({
    'SMILES':X_train['SMILES'].values.tolist(),
    'InChI':X_train['InChI'].values.tolist(),
    prop:y_train.values.tolist()
    })

test = pd.DataFrame({
    'SMILES':X_val['SMILES'].values.tolist(),
    'InChI':X_val['InChI'].values.tolist(),
    prop:y_val.values.tolist()
    })

# %% LogP Dist
fig = go.Figure()
fig.add_trace(go.Histogram(x=train[prop], name='train'))
fig.add_trace(go.Histogram(x=test[prop], name='test'))
fig.show()
# %%
train['LEN'] = train['SMILES'].apply(lambda x: len(x))
test['LEN'] = test['SMILES'].apply(lambda x: len(x))
train = train[train['LEN'] > 1].reset_index(drop=True)
test = test[test['LEN'] > 1].reset_index(drop=True)
train.to_csv(f"../Properties/{prop}/raw/train.csv", index=False)
test.to_csv(f"../Properties/{prop}/raw/test.csv", index=False)

# %%
# %%
