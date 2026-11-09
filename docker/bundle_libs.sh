#!/bin/bash
# Copy the non-system shared libraries a binary needs into DEST and point the binary's RUNPATH there, so the module or
# program runs in the official vLLM image, which lacks Mooncake's build dependencies (glog, gflags, yaml-cpp, ...).
#
#   bundle_libs.sh BINARY DEST RUNPATH [SEARCH_DIR]
#
# SEARCH_DIR (the build tree) supplies libraries the build made itself, such as Mooncake's own libasio.so, which no
# loader path reaches. Any library left unresolved (other than the driver's libcuda) is an error.
set -euo pipefail
binary=$1 dest=$2 runpath=$3 search=${4:-}
mkdir -p "$dest"
if [ -n "$search" ]; then
  for name in $(LD_LIBRARY_PATH="$dest" ldd "$binary" | awk '/=> not found/ {print $1}'); do
    found=$(find "$search" -name "$name" -print -quit)
    [ -n "$found" ] && cp -L "$found" "$dest/"
  done
fi
system_lib() {
  case $1 in
    libc.so*|libm.so*|libdl.so*|libpthread.so*|librt.so*|libgcc_s.so*|libstdc++.so*|ld-linux*|libibverbs.so*|\
    libnl-*|libnuma.so*|libcurl.so*|libssl.so*|libcrypto.so*|libz.so*|libnghttp2.so*|libidn2.so*|librtmp.so*|\
    libssh.so*|libpsl.so*|libgssapi*|libldap*|liblber*|libbrotli*|libunistring*|libgnutls*|libhogweed*|\
    libnettle*|libgmp*|libkrb5*|libk5crypto*|libcom_err*|libsasl2*|libp11-kit*|libtasn1*|libkeyutils*|libffi*|\
    libresolv*|libzstd.so*|libcuda.so*|libcudart.so*|libpython*) return 0 ;;
  esac
  return 1
}
for lib in $(LD_LIBRARY_PATH="$dest" ldd "$binary" | awk '/=>/ {print $3}'); do
  [ -f "$lib" ] || continue
  system_lib "$(basename "$lib")" && continue
  [ -f "$dest/$(basename "$lib")" ] || cp -L "$lib" "$dest/"
done
missing=$(LD_LIBRARY_PATH="$dest" ldd "$binary" | awk '/=> not found/ {print $1}' | grep -v '^libcuda\.so' || true)
[ -z "$missing" ] || { echo "unresolved libraries of $binary: $missing" >&2; exit 1; }
patchelf --set-rpath "$runpath" "$binary"
# RUNPATH is not inherited: each bundled library finds its own dependencies beside it.
for lib in "$dest"/*.so*; do
  [ -f "$lib" ] && patchelf --set-rpath '$ORIGIN' "$lib"
done
