#%%
from rdkit.Chem import PandasTools
import pandas as pd
# %%
props = ['LogP','MP','BP','WS','VP','HL','AOH','KOC','BCF','KM','KOA','BioHL']
for pr in props:
    tr = PandasTools.LoadSDF(f'../OPERA SDF Files/TR_{pr}.sdf',idName='ChemID',smilesName='SMILES')
    tst = PandasTools.LoadSDF(f'../OPERA SDF Files/TST_{pr}.sdf',idName='ChemID',smilesName='SMILES')
    cols = ["SMILES","InChI_Code_QSARr","NAME",tr.columns[-2]]
    if pr=='WS':
        cols = ["SMILES","InChI_Code_QSARr","NAME",tr.columns[-3]]
    train = tr[cols]
    test = tst[cols]
    train['LEN'] = train['SMILES'].apply(lambda x: len(x))
    test['LEN'] = test['SMILES'].apply(lambda x: len(x))
    train = train[train['LEN'] > 1].reset_index(drop=True)
    test = test[test['LEN'] > 1].reset_index(drop=True)
    train.to_csv(f"../OPERA Properties/{pr}/raw/train.csv", index=False)
    test.to_csv(f"../OPERA Properties/{pr}/raw/test.csv", index=False)
# %%
