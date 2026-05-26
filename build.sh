#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
PYTHON="python3"
GENERATOR="nmcm/generator"
PACKER="nmcm/packer"
PARSER="nmcm/parser"
DESKTOP_GEN="desktop_application/generated"
BUILD_DIR=".build"

# Fix: make nmcm_common visible to all sub-steps
export PYTHONPATH="$(pwd)/nmcm/common${PYTHONPATH:+:$PYTHONPATH}"

# ─── Helpers ──────────────────────────────────────────────────────────────────
log()  { echo "$*"; }
die()  { echo "✗ Error: $*" >&2; exit 1; }

need_rebuild() {
    local stamp="$1"; shift
    # Rebuild if stamp is missing; could be extended to check source mtimes
    [[ ! -f "$stamp" ]]
}

# ─── Steps ────────────────────────────────────────────────────────────────────
step_generator() {
    local stamp="$BUILD_DIR/generator.done"
    need_rebuild "$stamp" || { log "[1/4] ▶ nmcm_generator (up to date, skipping)"; return; }

    log "[1/4] ▶ nmcm_generator..."
    mkdir -p "$GENERATOR/output"
    (cd "$GENERATOR" && $PYTHON -m nmcm_generator) || die "nmcm_generator failed"
    touch "$stamp"
}

step_packer() {
    local stamp="$BUILD_DIR/packer.done"
    need_rebuild "$stamp" || { log "[2/4] ▶ nmcm_packer (up to date, skipping)"; return; }

    log "[2/4] ▶ nmcm_packer..."
    (cd "$PACKER" && $PYTHON -m nmcm_packer --mnist_dir ../generator/data) \
        || die "nmcm_packer failed"
    touch "$stamp"
}

step_parser() {
    local stamp="$BUILD_DIR/parser.done"
    need_rebuild "$stamp" || { log "[3/4] ▶ nmcm_parser (up to date, skipping)"; return; }

    log "[3/4] ▶ nmcm_parser..."
    mkdir -p "$PARSER/output"
    mkdir -p "$DESKTOP_GEN"
    (cd "$PARSER" && $PYTHON -m nmcm_parser \
        --model          ../generator/output/model.pth \
        --header_out_dir ../../desktop_application/generated \
        --hex_out_dir    output) \
        || die "nmcm_parser failed"
    touch "$stamp"
}

# ─── Desktop application ──────────────────────────────────────────────────────
BINARY="RelWithDebInfo/edgeist-desktop"
MODEL_HEX="nmcm/parser/output/model.hex"
MODEL_TRAINABLE_HEX="nmcm/parser/output/model_trainable.hex"
TRAIN_IMAGES="nmcm/packer/mnist_train_all_random.bin"
TEST_IMAGES="nmcm/packer/mnist_test_all_random.bin"

# ─── Commands ─────────────────────────────────────────────────────────────────
cmd_all() {
    mkdir -p "$BUILD_DIR"
    step_generator
    step_packer
    step_parser
    step_desktop
    log "✓ Pipeline abgeschlossen"
}

cmd_clean() {
    rm -rf "$BUILD_DIR"
    rm -rf "$GENERATOR/output"
    rm -rf "$PARSER/output"
    rm -rf "$DESKTOP_GEN"
    log "✓ Clean abgeschlossen"
}

cmd_force() {
    cmd_clean
    cmd_all
}

step_desktop() {
    log "[4/4] ▶ Building desktop application..."
    (cd desktop_application && make) || die "Desktop application build failed"
}

cmd_run() {
    [[ -f "$BINARY" ]]              || die "Binary not found: $BINARY — did you build the desktop application?"
    [[ -f "$MODEL_HEX" ]]          || die "Missing: $MODEL_HEX"
    [[ -f "$MODEL_TRAINABLE_HEX" ]] || die "Missing: $MODEL_TRAINABLE_HEX"
    [[ -f "$TRAIN_IMAGES" ]]        || die "Missing: $TRAIN_IMAGES"
    [[ -f "$TEST_IMAGES" ]]         || die "Missing: $TEST_IMAGES"

    log "▶ Running edgeist-desktop..."
    "$BINARY" \
        --model        "$MODEL_HEX" \
        --trainable    "$MODEL_TRAINABLE_HEX" \
        --train_images "$TRAIN_IMAGES" \
        --test_images  "$TEST_IMAGES"
}

# ─── Entry point ──────────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 [all|clean|force|run]"
    echo "  all    Build the full pipeline (default)"
    echo "  clean  Remove all build artefacts"
    echo "  force  Clean then rebuild everything"
    echo "  run    Run the desktop application with the generated model files"
    exit 1
}

case "${1:-all}" in
    all)   cmd_all   ;;
    clean) cmd_clean ;;
    force) cmd_force ;;
    run)   cmd_run   ;;
    *)     usage     ;;
esac
