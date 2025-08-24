# InsightX.spec
# PyInstaller spec file for robust portable build

from PyInstaller.utils.hooks import collect_all

# Collect torch & torchvision DLLs/data
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')
tv_datas, tv_binaries, tv_hiddenimports = collect_all('torchvision')

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=torch_binaries + tv_binaries,
    datas=torch_datas + tv_datas + [
        ('templates', 'templates'),
        ('static', 'static'),
        ('outputs/chestray_best.pt', 'outputs'),
    ],
    hiddenimports=[
        'pytorch_grad_cam',
        'sklearn.utils._cython_blas',
        'sklearn.utils._weight_vector',
        'sklearn.neighbors._partition_nodes',
    ] + torch_hiddenimports + tv_hiddenimports,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(pyz, a.scripts, name='InsightX', console=True)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=True, name='InsightX')
