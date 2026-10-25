#!/bin/sh
set -eu

registration_secret=/run/secrets/servicekit_registration_key

# Compose mounts the registration key as a secret so it is absent from
# `docker inspect .Config.Env`. ServiceKit still expects the documented env var.
if [ -z "${SERVICEKIT_REGISTRATION_KEY:-}" ] && [ -r "${registration_secret}" ]; then
    SERVICEKIT_REGISTRATION_KEY=$(sed -e 's/[[:space:]]*$//' "${registration_secret}")
    export SERVICEKIT_REGISTRATION_KEY
fi

if [ ! -w /work/data ]; then
    printf '%s\n' '{"level":"error","event":"startup_failed","reason":"/work/data is not writable"}' >&2
    exit 73
fi

if [ "$#" -eq 0 ]; then
    printf '%s\n' '{"level":"error","event":"startup_failed","reason":"no command supplied"}' >&2
    exit 64
fi

printf '%s\n' "{\"level\":\"info\",\"event\":\"service_start\",\"git_revision\":\"${GIT_REVISION:-unknown}\"}"
exec "$@"
