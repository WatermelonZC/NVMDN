"""Vectorized one-step EDD rollout. Same decisions as the scalar implementation."""
import numpy as np

def rollout_policy(o):
    mothers=o.get('mothers',list(range(1,5)));daughters=o.get('daughters',list(range(5,13)))
    vehicles=mothers+daughters;M=len(mothers);D=len(daughters);B=M*D;ix=np.arange(B)
    times=np.array([o['times'][v] for v in vehicles],dtype=float)
    pos=np.array([o['positions'][v] for v in vehicles],dtype=float)
    order=sorted(set(o['known'])-o['committed'],key=lambda i:(o['known'][i][5],i))
    rows=[np.asarray(o['known'][i],dtype=float) for i in order];actions=[]
    def candidates(t,p,row):
        join=np.abs(p[:,:M,None,:]-p[:,None,M:,:]).sum(-1)
        transport=np.abs(p[:,M:]-row[1:3]).sum(-1)+np.abs(row[1:3]-row[3:5]).sum()
        me=np.maximum(t[:,:M,None]+join/1.2,t[:,None,M:])+8.+transport[:,None,:]/1.2+30.+8.
        de=me+row[6];cost=.1*(join+transport[:,None,:])+.9*np.maximum(0,de-row[5])
        return me,de,cost
    def apply(t,p,row,m,d,me,de):
        t[ix,m]=me;t[ix,M+d]=de;p[ix,m]=row[3:5];p[ix,M+d]=row[3:5]
    mm=np.repeat(np.arange(M),D);dd=np.tile(np.arange(D),M)
    for k,(task,row) in enumerate(zip(order,rows)):
        me,de,cost=candidates(times[None,:],pos[None,:,:],row)
        me=me[0].reshape(-1);de=de[0].reshape(-1);cost=cost[0].reshape(-1)
        t=np.repeat(times[None,:],B,axis=0);p=np.repeat(pos[None,:,:],B,axis=0)
        value=cost.copy();apply(t,p,row,mm,dd,me,de)
        for future in rows[k+1:]:
            fm,fd,fc=candidates(t,p,future);best=fd.reshape(B,-1).argmin(axis=1);m=best//D;d=best%D
            value+=fc[ix,m,d];apply(t,p,future,m,d,fm[ix,m,d],fd[ix,m,d])
        pick=int(np.lexsort((dd,mm,de,value))[0]);m=int(mm[pick]);d=int(dd[pick])
        times[m]=me[pick];times[M+d]=de[pick];pos[m]=row[3:5];pos[M+d]=row[3:5]
        actions.append((task,mothers[m],daughters[d]))
    return actions
