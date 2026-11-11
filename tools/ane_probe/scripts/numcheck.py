import sys, numpy as np, coremltools as ct
pkg=sys.argv[1]; rng=np.random.default_rng(0)
m_cpu=ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_ONLY)
m_ane=ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
spec=m_cpu.get_spec()
x={i.name: rng.standard_normal([d for d in i.type.multiArrayType.shape]).astype(np.float32) for i in spec.description.input}
a=m_cpu.predict(x); b=m_ane.predict(x)
for k in a:
    d=np.abs(a[k]-b[k]); rel=np.linalg.norm(a[k]-b[k])/np.linalg.norm(a[k])
    print(f"{pkg.split('/')[-1]} {k}: relL2={rel:.2e} maxabs={d.max():.3e} nan={np.isnan(b[k]).sum()}")
