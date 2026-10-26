"""Export source, tests and documentation: python tools/package_project.py."""

import hashlib
from pathlib import Path
import shutil
import zipfile


PROJECT_NAME = 'depth_continuous_holo'
ROOT_FILES = (
    '.gitignore',
    'README.md',
    'THIRD_PARTY_NOTICES.md',
    'requirements.txt',
    'main_2d_diffdn.py',
    'test_2d_diffdn.py',
    'eval_2d_diffdn.py',
)
SOURCE_DIRS = ('datasets', 'models', 'tools', 'tests')


def collect_files(root):
    """Use a source whitelist and reject links to local/external files."""
    files = [root / name for name in ROOT_FILES]
    for name in SOURCE_DIRS:
        files.extend((root / name).rglob('*.py'))
    files.extend((root / 'docs').rglob('*.md'))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        for part in (path, *path.parents):
            if part == root:
                break
            if part.is_symlink():
                raise ValueError(f'Symlinks are not allowed in the export: {path}')
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def main():
    root = Path(__file__).resolve().parents[1]
    files = collect_files(root)
    output = root / 'dist'
    project = output / PROJECT_NAME
    archive = output / f'{PROJECT_NAME}.zip'
    checksum = output / f'{PROJECT_NAME}.zip.sha256'
    for path in (output, project, archive, checksum):
        if path.is_symlink():
            raise ValueError(f'Export path must not be a symlink: {path}')
    output.mkdir(exist_ok=True)
    if project.exists():
        shutil.rmtree(project)

    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as bundle:
        for source in files:
            relative = source.relative_to(root)
            target = project / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            content = source.read_bytes()
            target.write_bytes(content)
            # Stable metadata makes unchanged source produce the same archive.
            entry = zipfile.ZipInfo(f'{PROJECT_NAME}/{relative.as_posix()}')
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            bundle.writestr(entry, content, compresslevel=9)

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum.write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
    print(f'Exported {len(files)} files to {project.relative_to(root)}/')
    print(f'Archive: {archive.relative_to(root)} ({archive.stat().st_size:,} bytes)')
    print(f'SHA-256: {digest}')


if __name__ == '__main__':
    main()
