set -ex

cleanup() {
    set +e
    trap "" INT TERM
    kill -9 $server_pid
}
trap cleanup EXIT
trap "exit 24" INT TERM

BATCH_USE_KUBE_CONFIG=1 python batch/server.py &
server_pid=$!

until curl -fL 127.0.0.1:5000/jobs >/dev/null 2>&1
do
    sleep 1
done

POD_IP='127.0.0.1' BATCH_URL='http://127.0.0.1:5000' python -m unittest -v test/test_batch.py
