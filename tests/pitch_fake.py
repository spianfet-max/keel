import numpy as np, pandas as pd
rng=np.random.default_rng(0)
idx=pd.bdate_range("2004-01-01","2026-11-13")
def rw(start,vol,drift=0):
    r=rng.normal(drift/252,vol/np.sqrt(252),len(idx));return pd.Series(start*np.exp(np.cumsum(r)),index=idx)
class Fake:
    def __init__(s):
        s.cache={}
    def px(s,sym):
        if sym not in s.cache: s.cache[sym]=rw(100 if sym!="JPY=X" else 110,0.12,0.03)
        return s.cache[sym]
    def yld(s,k):
        if k not in s.cache: s.cache[k]=pd.Series(2+np.cumsum(rng.normal(0,0.05,len(idx))),index=idx)
        return s.cache[k]
    def curve(s,c): return {1.0:3.5,2.0:3.6,7.0:3.9,10.0:4.1}
    def rate(s,c):
        m=pd.date_range("2004-01-31","2026-10-31",freq="ME");return pd.Series({"USD":4.0,"JPY":0.5,"EUR":2.0}.get(c,3.0)+np.cumsum(rng.normal(0,.1,len(m))),index=m)
    def ppp_gap(s,c): return {"USD":0,"JPY":-0.3,"EUR":0.05}.get(c,0.1)
    def cot(s,k):
        d=pd.date_range("2006-01-03","2026-11-10",freq="W-TUE");return [{"d":str(x.date()),"pct":float(v)} for x,v in zip(d,np.cumsum(rng.normal(0,2,len(d))))]
    def mof(s):
        d=pd.date_range("2005-01-07","2026-11-07",freq="W-FRI");return [{"d":str(x.date()),"re":float(rng.normal(1000,3000)),"fe":float(rng.normal(0,3000)),"fb":float(rng.normal(0,3000))} for x in d]
    def cli(s,a):
        m=pd.date_range("2003-01-31","2026-09-30",freq="ME");return pd.Series(100+np.cumsum(rng.normal(0,.2,len(m))),index=m)
    def cpi(s,a):
        m=pd.date_range("2003-01-31","2026-09-30",freq="ME");return pd.Series(2+np.cumsum(rng.normal(0,.2,len(m))),index=m)
    def revisions(s,g): return {"name":"US","ja":"米国","n":60,"rev2m":0.01,"rev1m":0.005,"up_share":0.55,"dn_share":0.4,"net_analysts":0.1,"direction":1,"weeks_since_turn":None}
