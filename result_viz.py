#%%
import plotly.express as px
import pandas as pd
# %%
df = pd.read_csv('combined_result.csv')
# %%
px.bar(df.sort_values(by=['Test R2']), y='Properties', x='Test RMSE', color='Combination', barmode='group')
# %%

# %%
