#!/bin/bash
# Fix xtcocotools binary incompatibility with numpy 1.26.4.
# Run on server after git pull.
set -e

echo "=== Patching setuptools packaging bug ==="
PACKAGING_UTILS=$(python3 -c "import packaging.utils; print(packaging.utils.__file__)")
sed -i 's/value = name.lower()/value = (name or "").lower()/' "$PACKAGING_UTILS"
echo "Patched $PACKAGING_UTILS"

echo "=== Upgrading build tools ==="
pip install --upgrade setuptools wheel cython

echo "=== Rebuilding xtcocotools from source ==="
pip uninstall -y xtcocotools
pip install --no-binary :all: --force-reinstall xtcocotools

echo "=== Verify ==="
python3 -c "import xtcocotools; print('xtcocotools OK')"
python3 -c "import xtcocotools._mask as _mask; print('_mask OK')"

echo "Done."
