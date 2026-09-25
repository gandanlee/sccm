"""Post-process holo360d_megadepth:
1) Stratified cap of 3,000 per train scene / 4,000 for val (full version backed up in prep_scene_info_fullpairs/)
2) Remove all symlinks → copy the real files of the frames used (train/val images, whole test scenes)
Run in container: python3 postproc_full.py
"""
import os, shutil, numpy as np
from multiprocessing import Pool

DST   = os.environ.get('HOLO360D_DST',   'data/holo360d_megadepth')
RAW   = os.environ.get('HOLO360D_RAW',   'data/holo360d/train')
FUSED = os.environ.get('HOLO360D_FUSED', 'data/fused_megadepth/holo360d')
TRAIN=['Outdoor_003','Outdoor_005','Outdoor_006','Outdoor_007','Outdoor_008','Outdoor_010','Outdoor_012']
VAL='Outdoor_013'; TEST=['Outdoor_001','Outdoor_004','Outdoor_009','Outdoor_019']
SEED=0; DB=[2,4,7,10.01]; YB=[0,30,90,180.01]
BK=f'{DST}/prep_scene_info_fullpairs'; os.makedirs(BK,exist_ok=True)

def relang_all(P,pp):
    R=P[:,:3,:3]
    return np.array([np.degrees(np.arccos(np.clip((np.trace(R[i]@R[j].T)-1)/2,-1,1))) for i,j in pp])

def stratcap(d,y,ov,cap,rng):
    n=len(d)
    if n<=cap: return np.arange(n)
    db=np.digitize(d,DB[1:-1]); yb=np.digitize(y,YB[1:-1]); ob=(ov>=0.55).astype(int)
    bid=db*6+yb*2+ob; uniq=np.unique(bid); per=cap//len(uniq); sel=[]
    for u in uniq:
        idx=np.where(bid==u)[0]
        sel.append(idx[rng.permutation(len(idx))[:min(len(idx),per)]])
    sel=np.concatenate(sel)
    rest=np.setdiff1d(np.arange(n),sel); extra=cap-len(sel)
    if extra>0 and len(rest)>0: sel=np.concatenate([sel,rest[rng.permutation(len(rest))[:extra]]])
    return np.sort(sel)

def cp1(sd):
    s,dpath=sd
    if not os.path.exists(dpath): shutil.copy2(s,dpath)
    return 1

summary=[]
# ---- 1) cap + materialize train/val ----
for sc in TRAIN+[VAL]:
    npy=f'{DST}/prep_scene_info/holo360d_{sc}.npy'
    si=np.load(npy,allow_pickle=True).item()
    key='pairs_val' if sc==VAL else 'pairs_train'
    okey='overlaps_val' if sc==VAL else 'overlaps_train'
    pp=np.asarray(si[key]); ov=np.asarray(si[okey]); P=np.asarray(si['poses'],float)
    if not os.path.exists(f'{BK}/holo360d_{sc}.npy'):
        shutil.copy2(npy,f'{BK}/holo360d_{sc}.npy')
    C=np.stack([-P[i][:3,:3].T@P[i][:3,3] for i in range(len(P))])
    d=np.linalg.norm(C[pp[:,0]]-C[pp[:,1]],axis=1)
    y=relang_all(P,pp)
    cap=4000 if sc==VAL else 3000
    s=stratcap(d,y,ov,cap,np.random.default_rng(SEED))
    si[key]=pp[s].astype(np.int64); si[okey]=ov[s].astype(np.float64)
    np.save(npy[:-4],np.array(si,dtype=object))
    used=np.unique(pp[s].ravel())
    names=[si['image_paths'][f].split('/')[-1] for f in used]
    # images symlink → real copy
    imdir=f'{DST}/{sc}/images'
    if os.path.islink(imdir):
        os.unlink(imdir); os.makedirs(imdir)
    else: os.makedirs(imdir,exist_ok=True)
    jobs=[(f'{RAW}/{sc}/rgb/{nm}',f'{imdir}/{nm}') for nm in names]
    with Pool(12) as p: p.map(cp1,jobs,chunksize=16)
    summary.append((sc,'val' if sc==VAL else 'train',len(s),len(used)))
    print(f'[{sc}] cap {len(pp)}->{len(s)} pairs, images copied {len(used)}',flush=True)

# ---- 2) test scenes: dir symlink → real directory ----
for sc in TEST:
    npy=f'{DST}/prep_scene_info/holo360d_{sc}.npy'
    si=np.load(npy,allow_pickle=True).item()
    pp=np.asarray(si['pairs_test'])
    used=np.unique(pp.ravel())
    scdir=f'{DST}/{sc}'
    if os.path.islink(scdir): os.unlink(scdir)
    os.makedirs(f'{scdir}/images',exist_ok=True); os.makedirs(f'{scdir}/depths',exist_ok=True)
    jobs=[]
    for f in used:
        inm=si['image_paths'][f].split('/')[-1]; dnm=si['depth_paths'][f].split('/')[-1]
        jobs.append((f'{FUSED}/{sc}/images/{inm}',f'{scdir}/images/{inm}'))
        jobs.append((f'{FUSED}/{sc}/depths/{dnm}',f'{scdir}/depths/{dnm}'))
    with Pool(12) as p: p.map(cp1,jobs,chunksize=16)
    summary.append((sc,'test',len(pp),len(used)))
    print(f'[{sc}] materialized {len(used)} frames (jpg+h5), pairs={len(pp)}',flush=True)

print('\n===== FINAL =====')
tot={}
for sc,role,np_,nf in summary:
    print(f'{sc:14s} [{role:5s}] pairs={np_:6,d} frames={nf:5,d}')
    tot[role]=tot.get(role,0)+np_
print('Total:',tot)
# Check for leftover symlinks
left=[]
for root,dirs,files in os.walk(DST):
    for x in dirs+files:
        p=os.path.join(root,x)
        if os.path.islink(p): left.append(p)
    dirs[:]=[d for d in dirs if not os.path.islink(os.path.join(root,d))]
print('Remaining symlinks:',len(left),left[:5])
print('POSTPROC_DONE',flush=True)
