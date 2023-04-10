if command -v skopeo
then
    copy_image() {
        skopeo copy --override-os linux --override-arch amd64 docker://docker.io/$1 docker://$2
    }
else
    echo Could not find skopeo, falling back to docker which will be slower.
    copy_image() {
        docker pull $1
        docker tag $1 $2
        docker push $2
    }
fi
