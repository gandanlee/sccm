"""Full holo360d_megadepth build (B config: thin 0.5m, degcap 16, 2-10m, ov 0.3-0.8).
- train 7 scenes  -> pairs_train (corpus 'train')
- val  Outdoor_013 -> pairs_val  (corpus 'val')
- test 4 scenes (fused symlink + our pairs_test, strat cap 2000/scene)
Run: OPENCV_IO_ENABLE_OPENEXR=1 python3 scripts/data/build_holo360d.py
"""
import sys, os, time
import numpy as np, h5py, cv2, torch
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

RAW   = os.environ.get('HOLO360D_RAW',   'data/holo360d/train')
FUSED = os.environ.get('HOLO360D_FUSED', 'data/fused_megadepth/holo360d')
DST   = os.environ.get('HOLO360D_DST',   'data/holo360d_megadepth')
TRAIN=['Outdoor_003','Outdoor_005','Outdoor_006','Outdoor_007','Outdoor_008','Outdoor_010','Outdoor_012']
VAL='Outdoor_013'
TEST=['Outdoor_001','Outdoor_004','Outdoor_009','Outdoor_019']
# Distributed build: SCENES env specifies the train/val scenes; DO_TEST=1 processes the test scenes
if os.environ.get('SCENES'):
    _sel=os.environ['SCENES'].split(',')
    TRAIN=[s for s in TRAIN if s in _sel]
    if VAL not in _sel: VAL=None
if os.environ.get('DO_TEST','0')!='1': TEST=[]
THIN=0.5; DEG=16; SEED=0; H,W=448,896
DB=[2,4,7,10.01]; YB=[0,30,90,180.01]
M=np.array([[0,1,0],[0,0,1],[-1,0,0]],float)
NPROC=12
os.makedirs(f'{DST}/prep_scene_info',exist_ok=True)

def thin_idx(C):
    keep=[0]
    for i in range(1,len(C)):
        if np.linalg.norm(C[i]-C[keep[-1]])>=THIN: keep.append(i)
    return np.array(keep)

def degcap(pairs,rng):
    cnt={}; out=[]
    for k in rng.permutation(len(pairs)):
        i,j=pairs[k]
        if cnt.get(i,0)<DEG and cnt.get(j,0)<DEG:
            out.append(k); cnt[i]=cnt.get(i,0)+1; cnt[j]=cnt.get(j,0)+1
    return np.sort(np.array(out))

def stratcap(d,y,ov,cap,rng):
    n=len(d)
    if n<=cap: return np.arange(n)
    db=np.digitize(d,DB[1:-1]); yb=np.digitize(y,YB[1:-1]); ob=(ov>=0.55).astype(int)
    bid=db*6+yb*2+ob
    uniq=np.unique(bid); per=cap//len(uniq); sel=[]
    for u in uniq:
        idx=np.where(bid==u)[0]
        sel.append(idx[rng.permutation(len(idx))[:min(len(idx),per)]])
    sel=np.concatenate(sel)
    rest=np.setdiff1d(np.arange(n),sel); extra=cap-len(sel)
    if extra>0 and len(rest)>0: sel=np.concatenate([sel,rest[rng.permutation(len(rest))[:extra]]])
    return np.sort(sel)

# ---------- depth conversion worker ----------
def conv_depth(args):
    sc,nm=args
    out=f'{DST}/{sc}/depths/{nm}.h5'
    if os.path.exists(out): return 0
    try:
        mesh=cv2.imread(f'{RAW}/{sc}/depth/mesh_depth/{nm}.exr',cv2.IMREAD_UNCHANGED)
        mesh=mesh if mesh.ndim==2 else mesh[...,0]
        r=cv2.resize(np.asarray(mesh,np.float32),(1440,720),interpolation=cv2.INTER_LINEAR)
        r[~np.isfinite(r)]=0
        mk=cv2.imread(f'{RAW}/{sc}/mask/{nm}.jpg',cv2.IMREAD_GRAYSCALE)
        mr=cv2.resize((mk>127).astype(np.float32),(1440,720),interpolation=cv2.INTER_LINEAR)
        r[mr<1.0]=0
        with h5py.File(out,'w') as h: h.create_dataset('depth',data=r,compression='gzip',compression_opts=3)
        return 1
    except Exception as e:
        print('DEPTH_FAIL',sc,nm,e,flush=True); return -1

# ---------- overlap worker ----------
_G={}
def ov_init(sc,names,poses):
    torch.set_num_threads(1)
    _G['sc']=sc; _G['names']=names; _G['poses']=poses
def ov_one(pair):
    from sccm.utils.utils_sphere import get_gt_warp_erp
    a,b=pair; sc=_G['sc']; names=_G['names']; poses=_G['poses']
    try:
        with h5py.File(f'{DST}/{sc}/depths/{names[a]}.h5','r') as h: dA=torch.tensor(h['depth'][()])
        with h5py.File(f'{DST}/{sc}/depths/{names[b]}.h5','r') as h: dB=torch.tensor(h['depth'][()])
        T=torch.tensor(poses[b]@np.linalg.inv(poses[a]),dtype=torch.float32)
        gw,gp=get_gt_warp_erp(dA[None],dB[None],T[None],H=H,W=W)
        return float((gp[0]>0.99).float().mean())
    except Exception as e:
        print('OV_FAIL',sc,a,b,e,flush=True); return -1.0

def relang(Ri,Rj):
    return np.degrees(np.arccos(np.clip((np.trace(Ri@Rj.T)-1)/2,-1,1)))

# ================= train/val scenes =================
for sc in TRAIN+([VAL] if VAL else []):
    t0=time.time()
    lines=open(f'{RAW}/{sc}/poses/pose.txt').read().strip().split('\n')[1:]
    names=[l.split()[0].rsplit('.',1)[0] for l in lines]
    XYZ=np.array([[float(v) for v in l.split()[1:4]] for l in lines])
    RR=np.array([[float(v) for v in l.split()[4:13]] for l in lines]).reshape(-1,3,3)
    n=len(names)
    poses=np.zeros((n,4,4)); poses[:,3,3]=1
    for i in range(n):
        Rf=M@RR[i]; poses[i,:3,:3]=Rf; poses[i,:3,3]=-Rf@XYZ[i]
    keep=thin_idx(XYZ)
    ii,jj=np.triu_indices(len(keep),k=1)
    d=np.linalg.norm(XYZ[keep][ii]-XYZ[keep][jj],axis=1)
    m=(d>=2)&(d<=10)
    pp=np.stack([keep[ii[m]],keep[jj[m]]],1)
    rng=np.random.default_rng(SEED)
    cand=pp[degcap(pp,rng)]
    frames=np.unique(cand.ravel())
    print(f'[{sc}] frames={n} thin={len(keep)} cand={len(cand)} uframes={len(frames)}',flush=True)
    os.makedirs(f'{DST}/{sc}/depths',exist_ok=True)
    if not os.path.lexists(f'{DST}/{sc}/images'):
        os.symlink(f'{RAW}/{sc}/rgb', f'{DST}/{sc}/images')
    with Pool(NPROC) as p:
        r=p.map(conv_depth,[(sc,names[f]) for f in frames],chunksize=8)
    print(f'[{sc}] depths: new={sum(1 for x in r if x==1)} skip={sum(1 for x in r if x==0)} fail={sum(1 for x in r if x<0)} ({time.time()-t0:.0f}s)',flush=True)
    with Pool(NPROC,initializer=ov_init,initargs=(sc,names,poses)) as p:
        ovs=np.array(p.map(ov_one,[tuple(x) for x in cand],chunksize=4))
    good=(ovs>=0.3)&(ovs<=0.8)
    sel=cand[good]; selov=ovs[good]
    prep=dict(scene=f'holo360d_{sc}',
              scene_corpus=('val' if sc==VAL else 'train'),
              poses=poses, intrinsics=np.tile(np.eye(3),(n,1,1)),
              image_paths=np.array([f'{sc}/images/{nm}.jpg' for nm in names]),
              depth_paths=np.array([f'{sc}/depths/{nm}.h5' for nm in names]),
              pairs_train=np.zeros((0,2),np.int64), overlaps_train=np.array([],np.float64),
              pairs_val=np.zeros((0,2),np.int64),   overlaps_val=np.array([],np.float64),
              pairs_test=np.zeros((0,2),np.int64),  overlaps_test=np.array([],np.float64))
    if sc==VAL: prep['pairs_val']=sel.astype(np.int64); prep['overlaps_val']=selov.astype(np.float64)
    else:       prep['pairs_train']=sel.astype(np.int64); prep['overlaps_train']=selov.astype(np.float64)
    np.save(f'{DST}/prep_scene_info/holo360d_{sc}', np.array(prep,dtype=object))
    print(f'[{sc}] DONE pairs={len(sel)} (ov med {np.median(selov):.3f}) total {time.time()-t0:.0f}s',flush=True)

# ================= test scenes (reuse fused) =================
for sc in TEST:
    si=np.load(f'{FUSED}/prep_scene_info/holo360d_{sc}.npy',allow_pickle=True).item()
    P=np.asarray(si['poses'],float); n=len(P)
    C=np.stack([-P[i][:3,:3].T@P[i][:3,3] for i in range(n)])
    keep=set(thin_idx(C).tolist())
    pp=np.asarray(si['pairs_train']); ov=np.asarray(si['overlaps_train'])
    d=np.linalg.norm(C[pp[:,0]]-C[pp[:,1]],axis=1)
    m=(d>=2)&(d<=10)&(ov>=0.3)&(ov<=0.8)&np.array([(i in keep) and (j in keep) for i,j in pp])
    idx=np.where(m)[0]
    rng=np.random.default_rng(SEED)
    kept=idx[degcap(pp[idx],rng)]
    Rw=P[:,:3,:3]
    y=np.array([relang(Rw[i],Rw[j]) for i,j in pp[kept]])
    s=kept[stratcap(d[kept],y,ov[kept],2000,rng)]
    out=dict(si); out['scene_corpus']='test'
    out['pairs_train']=np.zeros((0,2),np.int64); out['overlaps_train']=np.array([],np.float64)
    out['pairs_val']=np.zeros((0,2),np.int64);   out['overlaps_val']=np.array([],np.float64)
    out['pairs_test']=pp[s].astype(np.int64);    out['overlaps_test']=ov[s].astype(np.float64)
    np.save(f'{DST}/prep_scene_info/holo360d_{sc}', np.array(out,dtype=object))
    if not os.path.lexists(f'{DST}/{sc}'):
        os.symlink(f'../fused_megadepth/holo360d/{sc}', f'{DST}/{sc}')
    print(f'[{sc}] TEST pairs={len(s)}',flush=True)

print('BUILD_FULL_DONE',flush=True)
