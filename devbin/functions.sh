kfind1() {
    # kfind1 app [namespace]
    #
    # Print one pod from namespace `namespace` whose app label matches `app`.
    #
    # Example:
    #
    #     # kfind1 batch
    #     batch-8c6c74ffd-d498g
    kubectl -n ${2:-default} get pods -l app="$1" --no-headers | grep Running | awk '{ print $1 }' | head -n 1
}

klf() {
    # klf app [namespace]
    #
    # Print the logs in real time from some matching pod.
    #
    # Example:
    #
    #     # klf batch
    #     ... logs are streamed to your computer ...
    kubectl -n ${2:-default} logs -f "$(kfind1 "$1" "$2")"
}

kssh() {
    # kssh app [namespace]
    #
    # Connect to a bash session on some matching pod.
    #
    # Example:
    #
    #     # kssh admin-pod
    #     root@admin-pod-5d77d69445-86m2h:/#
    kubectl -n ${2:-default} exec -it "$(kfind1 "$1" "$2")" ${3:+--container="$3"} -- ${KSSH_SHELL:-/bin/sh}
}

klog() {
    # klog app [namespace]
    #
    # Print all available non-healthcheck, non-metric logs for *all* matching pods in chronological
    # order.
    #
    # Example:
    #
    #     # klog batch dking
    #     ... logs ...
    dir=$(mktemp -d)
    mkdir -p $dir
    for x in $(kubectl get pods -l app="${1}" -n "${2}" | tail -n +2 | awk '{print $1}')
    do
        kubectl logs --tail=999999999999999 $x -n "${2}" --all-containers \
            | grep -Ev 'healthcheck|metrics' \
                   > $dir/$x &
    done
    wait
    cat $dir/* | sort
}

kjlog() {
    # kjlog app [namespace]
    #
    # Print all available non-healthcheck, non-metric logs for *all* matching pods in chronological
    # order. Non-JSON logs are elided. Not all fields are shown.
    #
    # Example:
    #
    #     # kjlog batch dking
    #     ...
    #     {"asctime":"2021-02-12 16:58:49,540","pod":"batch-8c6c74ffd-d498g","x_real_ip":"35.187.114.193","connection_id":null,"exc_info":null,"message":"https GET / done in 0.0015127049991860986s: 302"}
    #     {"asctime":"2021-02-12 16:58:57,850","pod":"batch-8c6c74ffd-lzfdh","x_real_ip":"104.197.30.241","connection_id":null,"exc_info":null,"message":"https GET / done in 0.0010090250007124268s: 302"}
    #     {"asctime":"2021-02-12 16:59:11,747","pod":"batch-8c6c74ffd-lzfdh","x_real_ip":"18.31.31.24","connection_id":null,"exc_info":null,"message":"https GET /api/v1alpha/batches/182764 done in 0.04281349099983345s: 200"}
    #     {"asctime":"2021-02-12 16:59:12,907","pod":"batch-8c6c74ffd-jmhmq","x_real_ip":"35.198.194.122","connection_id":null,"exc_info":null,"message":"https GET / done in 0.0008628759969724342s: 302"}
    dir=$(mktemp -d)
    mkdir -p $dir
    for x in $(kubectl get pods -l app="${1}" -n "${2}" | tail -n +2 | awk '{print $1}')
    do
        kubectl logs --tail=999999999999999 $x -n "${2}" --all-containers \
            | grep -Ev 'healthcheck|metrics' \
            | grep '^{' \
            | jq -c '{time: (if .asctime then .asctime else .["@timestamp"] end), pod: "'$x'", x_real_ip, connection_id, exc_info, message}' \
                 > $dir/$x &
    done
    wait
    cat $dir/* | sort | jq -c '{time, pod, x_real_ip, connection_id, exc_info, message}'
}

kjlogs() {
    # kjlogs namespace app1 [app2 ...]
    #
    # Print all available non-healthcheck, non-metric logs for pods from any of the specified apps
    # in chronological order. Non-JSON logs are elided. Not all fields are shown.
    #
    # Example:
    #
    #     # kjlogs default batch batch-driver
    #     ...
    #     {"asctime":"2021-02-12 17:01:53,832","pod":"batch-8c6c74ffd-lzfdh","x_real_ip":null,"connection_id":null,"exc_info":null,"message":"https GET /api/v1alpha/batches/182767 done in 0.0401439390006999s: 200"}
    #     {"asctime":"2021-02-12 17:01:55,553","pod":"batch-driver-6748cd87f9-kdjv8","x_real_ip":null,"connection_id":null,"exc_info":null,"message":"marking job (182768, 140) complete new_state Success"}
    dir=$(mktemp -d)
    mkdir -p $dir
    namespace="${1}"
    shift
    for app in $*
    do
        for x in $(kubectl get pods -l app="${app}" -n "${namespace}" | tail -n +2 | awk '{print $1}')
        do
            kubectl logs --tail=999999999999999 $x -n "${namespace}" --all-containers \
                | grep -Ev 'healthcheck|metrics' \
                | grep '^{' \
                | jq -c '{time: (if .asctime then .asctime else .["@timestamp"] end), pod: "'$x'", x_real_ip, connection_id, exc_info, message}' \
                     > $dir/$x &
        done
    done
    wait
    cat $dir/* | sort | jq -c '{time, pod, x_real_ip, connection_id, exc_info, message}'
}

knodes() {
    # knodes
    #
    # Print every kubernetes node and the non-kube-system pods resident on that node.
    #
    # Example:
    #
    #     # knodes
    #     ...
    #     ====== gke-vdc-preemptible-pool-3-770fae6d-zl4l =====
    #     NAMESPACE     NAME                                                  READY   STATUS    RESTARTS   AGE   IP             NODE                                       NOMINATED NODE   READINESS GATES
    #     default       address-6b465c7594-bk9lb                              1/1     Running   0          23m   10.32.34.39    gke-vdc-preemptible-pool-3-770fae6d-zl4l   <none>           <none>
    for i in $(kubectl get nodes | awk '{print $1}')
    do
        echo "====== $i ====="
        kubectl get pods --all-namespaces -o wide --field-selector spec.nodeName=$i \
          | awk '$1 != "kube-system" { print }'
    done
}

download-secret() {
    # download-secret secret-name namespace
    #
    # Download a secret, base64 unencode the contents, and write into files on your computer. Use
    # `popd` to return to your previous working directory.
    #
    # Example:
    #
    #     # download-secret ssl-config-batch dking
    #     /var/folders/cq/p_l4jm3x72j7wkxqxswccs180000gq/T/tmp.NTb5sZMX ~/projects/hail
    #     (base) # ls contents
    #     batch-cert.pem			batch-incoming-store.jks	batch-incoming.pem		batch-key-store.p12		batch-key.pem			batch-outgoing-store.jks	batch-outgoing.pem		ssl-config.json
    #     (base) # ls
    #     contents	secret.json
	  name=$1
	  namespace=${2:-default}
	  pushd $(mktemp -d)
	  kubectl get secret $name --namespace $namespace -o json > secret.json
	  mkdir contents
	  for field in $(jq -r  '.data | keys[]' secret.json)
	  do
		    jq -r '.data["'$field'"]' secret.json | base64 --decode > contents/$field
	  done
}

upload-secret() {
    # upload-secret
    #
    # Upload a secret that has been previously downloaded using download-secret. If you intend to
    # upload to a different namespace, ensure you've also modified secret.json.
	  name=$(jq -r '.metadata.name' secret.json)
	  namespace=$(jq -r '.metadata.namespace' secret.json)
	  kubectl create secret generic $name --namespace $namespace $(for i in $(ls contents); do echo "--from-file=contents/$i" ; done) --save-config --dry-run=client -o yaml \
	      | kubectl apply -f -
}

download-configmap() {
    # download-configmap configmap-name namespace
    #
    # Download a configmap and write into files on your computer. Use
    # `popd` to return to your previous working directory.
    #
    # Example:
    #
    #     # download-configmap gateway-xds-config
    #     /var/folders/cq/p_l4jm3x72j7wkxqxswccs180000gq/T/tmp.NTb5sZMX ~/projects/hail
    #     # ls
    #     contents	configmap.json
    #     # ls contents
    #     rds.yaml			cds.yaml
	  name=$1
	  namespace=${2:-default}
	  pushd $(mktemp -d)
	  kubectl get configmap $name --namespace $namespace -o json > configmap.json
	  mkdir contents
	  for field in $(jq -r  '.data | keys[]' configmap.json)
	  do
		    jq -r '.data["'$field'"]' configmap.json > contents/$field
	  done
}

get_global_config_field() {
    kubectl -n ${2:-default} get secret global-config --template={{.data.$1}} | base64 --decode
}

gcpsetcluster() {
    if [ -z "$1" ]; then
        echo "Usage: gcpsetcluster <PROJECT>"
        return
    fi

    gcloud config set project $1
    gcloud container clusters get-credentials --zone us-central1-a vdc
}

azsetcluster() {
    if [ -z "$1" ]; then
        echo "Usage: azsetcluster <RESOURCE_GROUP>"
        return
    fi

    RESOURCE_GROUP=$1
    az aks get-credentials --name vdc --resource-group $RESOURCE_GROUP
    az acr login --name $(get_global_config_field docker_prefix)
}

azsshworker() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Usage: azsshworker <RESOURCE_GROUP> <WORKER_NAME>"
        return
    fi

    RESOURCE_GROUP=$1
    WORKER_NAME=$2
    SSH_PRIVATE_KEY_PATH=$3

    worker_ip=$(az vm list-ip-addresses -g $RESOURCE_GROUP -n $WORKER_NAME \
        | jq -jr '.[0].virtualMachine.network.publicIpAddresses[0].ipAddress')

    ssh -i ~/.ssh/batch_worker_ssh_rsa batch-worker@$worker_ip
}

confirm() {
    printf "$1\n"
    read -r -p "Are you sure? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            true
            ;;
        *)
            false
            ;;
    esac
}
