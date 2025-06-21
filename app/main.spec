# main.spec

# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets', 'assets'),
        ('logic', 'logic'),
        ('ui', 'ui'),
        ('run_model', 'run_model'),
        ('.venv/Lib/site-packages/mediapipe/modules/pose_landmark', 'mediapipe/modules/pose_landmark'),
        ('.venv/Lib/site-packages/mediapipe/modules/pose_detection', 'mediapipe/modules/pose_detection'),
        ('.venv/Lib/site-packages/vosk/libvosk.dll', 'vosk')
    ],
    hiddenimports=['vosk', 'tensorboard', 'lightning_utilities'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Shadow Control',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,               # --noconsole
    icon='icon.ico'              # --icon=icon.ico
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Shadow'
)
