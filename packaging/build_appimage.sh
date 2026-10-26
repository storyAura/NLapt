#!/usr/bin/env bash
# Build a Linux AppImage for the NLapt GUI.
#
#   bash packaging/build_appimage.sh
#
# Counterpart of packaging/build.ps1 (Windows): it runs the same PyInstaller spec
# (packaging/nlapt.spec) to get the onedir bundle under dist/NLapt, then wraps that
# folder into an AppImage with appimagetool. The result is
# dist/NLapt-<version>-x86_64.AppImage.
#
# The spec already fails loudly when a runtime dependency is missing; install the
# build deps first (a venv is fine):
#
#   pip install -e .[gui,images,llm,local] && pip install pyinstaller
#
# appimagetool is downloaded once into dist/cache and reused afterwards.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

python="${PYTHON:-python3}"
if ! "$python" -c "import PyInstaller" >/dev/null 2>&1; then
  echo "PyInstaller is not importable by '$python'. Install it first:" >&2
  echo "  $python -m pip install pyinstaller" >&2
  exit 1
fi

version="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)"
if [ -z "$version" ]; then
  echo "Could not read the version from pyproject.toml" >&2
  exit 1
fi

dist="$root/dist"
appdir="$root/build/appimage/NLapt.AppDir"
bundle="$dist/NLapt"
icon="$root/packaging/icon.png"

if [ ! -f "$icon" ]; then
  echo "Missing $icon (the Linux icon, derived from packaging/icon.ico)" >&2
  exit 1
fi

echo "Building NLapt $version with PyInstaller..."
"$python" -m PyInstaller packaging/nlapt.spec --noconfirm
if [ ! -x "$bundle/NLapt" ]; then
  echo "PyInstaller did not produce $bundle/NLapt" >&2
  exit 1
fi

# appimagetool: cached in dist/cache, downloaded only once.
tool="$dist/cache/appimagetool-x86_64.AppImage"
if [ ! -x "$tool" ]; then
  mkdir -p "$dist/cache"
  echo "Downloading appimagetool (once; later runs reuse dist/cache)"
  if ! curl -fL --retry 3 -o "$tool.part" \
    https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage; then
    rm -f "$tool.part"
    echo "Downloading appimagetool failed. Put a copy at $tool and retry." >&2
    exit 1
  fi
  mv "$tool.part" "$tool"
  chmod +x "$tool"
fi

# Assemble the AppDir. PyInstaller's onedir bundle keeps its libraries next to the
# executable, so the whole folder goes into usr/bin.
echo "Assembling AppDir..."
rm -rf "$appdir"
mkdir -p "$appdir/usr/bin" "$appdir/usr/share/applications" \
  "$appdir/usr/share/icons/hicolor/256x256/apps"
cp -a "$bundle/." "$appdir/usr/bin/"
cp "$icon" "$appdir/nlapt.png"
cp "$icon" "$appdir/usr/share/icons/hicolor/256x256/apps/nlapt.png"
cp "$icon" "$appdir/.DirIcon"

cat >"$appdir/nlapt.desktop" <<'DESKTOP'
[Desktop Entry]
Type=Application
Name=NLapt
GenericName=Caption editor
Comment=Edit and batch-modify natural-language image captions
Exec=NLapt %F
Icon=nlapt
Categories=Graphics;Utility;
Keywords=caption;tag;dataset;lora;
Terminal=false
MimeType=inode/directory;
DESKTOP
cp "$appdir/nlapt.desktop" "$appdir/usr/share/applications/nlapt.desktop"

cat >"$appdir/AppRun" <<'APPRUN'
#!/bin/sh
# The bundle looks for its libraries next to the executable, so start from usr/bin.
here="$(dirname "$(readlink -f "$0")")"
cd "$here/usr/bin"
exec ./NLapt "$@"
APPRUN
chmod +x "$appdir/AppRun"

echo "Building the AppImage..."
name="NLapt-$version-x86_64.AppImage"
ARCH=x86_64 "$tool" "$appdir" "$dist/$name"

echo
echo "Done."
echo "  AppImage: $dist/$name"
