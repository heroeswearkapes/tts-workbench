#!/bin/sh
set -eu

for directory in /models /voices /input /output /config; do
    mkdir -p "$directory"
    if [ ! -w "$directory" ]; then
        echo "TTS Workbench storage path is not writable: $directory" >&2
        exit 1
    fi
done

if [ "$#" -eq 0 ]; then
    set -- tts-workbench
fi

if [ "$1" = "tts-workbench" ]; then
    echo "TTS Workbench container startup diagnostics"
    python -m tts_workbench.device_diagnostics
fi

exec "$@"
