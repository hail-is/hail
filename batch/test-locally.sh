set -ex

. activate hail-batch

cleanup() {
    set +e
    trap "" INT TERM
    kill -9 $server_pid
}
trap cleanup EXIT
trap "exit 24" INT TERM

python -c 'import batch.server; batch.server.serve()' &
server_pid=$!

until curl -fL 127.0.0.1:5000/jobs >/dev/null 2>&1
do
    ((tries++)) && [ $tries -lt 30 ]
    sleep 1
done

POD_IP='127.0.0.1' BATCH_URL='http://127.0.0.1:5000' python -m unittest -v test/test_batch.py
