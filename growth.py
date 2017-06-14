import pandas as pd
import numpy as np
import statsmodels.api as sm

# Growth data from this page:
# http://www.bristol.ac.uk/cmm/learning/support/datasets/

colspecs = [(0, 4), (4, 7), (7, 12), (12, 16), (16, 17)]

df = pd.read_fwf("data/growth/ASIAN.DAT", colspecs=colspecs, header=None)
df.columns = ["Id", "Age", "Weight", "BWeight", "Gender"]
df["Female"] = 1*(df.Gender == 2)
df = df.dropna()

df["LogWeight"] = np.log(df.Weight) / np.log(2)
df["LogBWeight"] = np.log(df.BWeight) / np.log(2)

model0 = sm.GLM.from_formula("Weight ~ Age + BWeight + Female", data=df)
rslt0 = model0.fit()

model1 = sm.GEE.from_formula("Weight ~ Age + BWeight + Female", groups="Id", data=df)
rslt1 = model1.fit()

model2 = sm.GEE.from_formula("LogWeight ~ Age + LogBWeight + Female", groups="Id", data=df)
rslt2 = model2.fit()

model3 = sm.GEE.from_formula("LogWeight ~ bs(Age, 4) + LogBWeight + Female", groups="Id", data=df)
rslt3 = model3.fit()

model4 = sm.GEE.from_formula("LogWeight ~ bs(Age, 4) + LogBWeight*Female", groups="Id", data=df)
rslt4 = model4.fit()

model5 = sm.GEE.from_formula("LogWeight ~ bs(Age, 4) + LogBWeight + Female", groups="Id",
                             cov_struct=sm.cov_struct.Exchangeable(), data=df)
rslt5 = model5.fit()

model6 = sm.GEE.from_formula("LogWeight ~ bs(Age, 4) + LogBWeight + Female", groups="Id",
                             cov_struct=sm.cov_struct.Exchangeable(), data=df)
rslt6 = model6.fit(cov_type="naive")
