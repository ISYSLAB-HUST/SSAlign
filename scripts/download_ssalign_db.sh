#!/bin/sh

set -eu

usage() {
    cat <<'EOF'
Download a prebuilt SSAlign database file.

Usage:
  ./scripts/download_ssalign_db.sh [DATABASE] [DIMENSION] [OUTPUT_DIR]

Arguments:
  DATABASE    Database name (default: swissprot)
  DIMENSION   Embedding dimension (default: 512)
  OUTPUT_DIR  Destination directory (default: models/SSAlignDB/SwissProt)

Currently available:
  swissprot 512
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
        filename="SwissProt_IndexFlatIP_512_faiss.index"
        default_url="http://bioinfo.isyslab.info/ssalign/download/external/swissprot/SSAlignDB/file/$filename"
        default_size=1110790189
        ;;
    *)
        printf 'Error: unsupported database/dimension: %s %s\n' "$database" "$dimension" >&2
        printf 'Run %s --help to list available downloads.\n' "$0" >&2
        exit 2
        ;;
esac

output_dir=${3:-"$repo_root/models/SSAlignDB/$database_dir"}
url=${SSALIGN_DOWNLOAD_URL:-$default_url}
expected_size=${SSALIGN_EXPECTED_SIZE:-$default_size}
destination="$output_dir/$filename"
partial="$destination.part"

mkdir -p "$output_dir"

file_size() {
    wc -c < "$1" | tr -d '[:space:]'
}

if [ -f "$destination" ]; then
    current_size=$(file_size "$destination")
    if [ "$current_size" = "$expected_size" ]; then
        printf 'Database file already exists and has the expected size:\n  %s\n' "$destination"
        exit 0
    fi
    printf 'Error: existing file has %s bytes; expected %s.\n' "$current_size" "$expected_size" >&2
    printf 'Remove the file and run the command again:\n  %s\n' "$destination" >&2
    exit 1
fi

printf 'Downloading %s (%s dimensions)\n' "$database_dir" "$dimension"
printf 'Source: %s\n' "$url"
printf 'Destination: %s\n' "$destination"

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
    printf 'Error: downloaded %s bytes; expected %s.\n' "$actual_size" "$expected_size" >&2
    printf 'The partial file was kept so the download can be resumed:\n  %s\n' "$partial" >&2
    exit 1
fi

mv "$partial" "$destination"
printf 'Download complete:\n  %s\n' "$destination"

