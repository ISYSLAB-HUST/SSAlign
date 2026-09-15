#!/bin/sh

set -eu

usage() {
    cat <<'EOF'
Download a prebuilt SSAlign database bundle.

Usage:
  ./scripts/download_ssalign_db.sh [DATABASE] [DIMENSION] [OUTPUT_DIR]

Arguments:
  DATABASE    Database name (default: swissprot)
  DIMENSION   Embedding dimension (default: 512)
  OUTPUT_DIR  Destination directory (default: models/SSAlignDB/SwissProt)

Currently available:
  swissprot 512

The SwissProt bundle contains the FAISS index, the protein ID/3Di-sequence
mapping, and the two whitening-transform files required for local search.
EOF
}

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
esac

database=$(printf '%s' "${1:-swissprot}" | tr '[:upper:]' '[:lower:]')
dimension=${2:-512}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)

case "$database:$dimension" in
    swissprot:512)
        database_dir="SwissProt"
        default_base_url="http://bioinfo.isyslab.info/ssalign/download/external/swissprot/SSAlignDB/file"
        ;;
    *)
        printf 'Error: unsupported database/dimension: %s %s\n' "$database" "$dimension" >&2
        printf 'Run %s --help to list available downloads.\n' "$0" >&2
        exit 2
        ;;
esac

output_dir=${3:-"$repo_root/models/SSAlignDB/$database_dir"}
base_url=${SSALIGN_DOWNLOAD_BASE_URL:-$default_base_url}
manifest=${SSALIGN_DOWNLOAD_MANIFEST:-}

mkdir -p "$output_dir"

file_size() {
    wc -c < "$1" | tr -d '[:space:]'
}

download_file() {
    filename=$1
    expected_size=$2
    url="$base_url/$filename"
    destination="$output_dir/$filename"
    partial="$destination.part"

    if [ -f "$destination" ]; then
        current_size=$(file_size "$destination")
        if [ "$current_size" = "$expected_size" ]; then
            printf 'Already installed: %s\n' "$destination"
            return
        fi
        printf 'Error: %s has %s bytes; expected %s.\n' \
            "$destination" "$current_size" "$expected_size" >&2
        printf 'Remove the incomplete file and run the command again.\n' >&2
        exit 1
    fi

    printf 'Downloading: %s\n' "$filename"
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --retry-delay 2 \
            --continue-at - --output "$partial" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget --continue --output-document="$partial" "$url"
    else
        printf 'Error: curl or wget is required.\n' >&2
        exit 1
    fi

    actual_size=$(file_size "$partial")
    if [ "$actual_size" != "$expected_size" ]; then
        printf 'Error: downloaded %s bytes for %s; expected %s.\n' \
            "$actual_size" "$filename" "$expected_size" >&2
        printf 'The .part file was retained for resuming the download.\n' >&2
        exit 1
    fi

    mv "$partial" "$destination"
    printf 'Installed: %s\n' "$destination"
}

printf 'Installing %s database bundle (%s-dimensional index) in:\n  %s\n' \
    "$database_dir" "$dimension" "$output_dir"

if [ -n "$manifest" ]; then
    while read -r filename expected_size; do
        [ -n "$filename" ] || continue
        download_file "$filename" "$expected_size"
    done < "$manifest"
else
    while read -r filename expected_size; do
        download_file "$filename" "$expected_size"
    done <<'EOF'
SwissProt_IndexFlatIP_512_faiss.index 1110790189
SwissProt_id_Seq.npz 407602632
SwissProt_whitening_W.npy 13107328
SwissProt_whitening_mu.npy 10368
EOF
fi

printf 'SwissProt database installation complete.\n'
