if command -v skopeo
then
    copy_image() {
        src=$1
        # Check if source already has a registry prefix
        # A registry is indicated by the first part (before first '/') containing a '.' or ':'
        # Examples: "docker.io/ubuntu", "us-central1-docker.pkg.dev/...", "localhost:5000/image"
        if [[ "$src" == *"/"* ]]
        then
            # Extract the first part (registry/namespace)
            first_part="${src%%/*}"
            # If first part contains '.' or ':', it's a registry
            if [[ "$first_part" == *"."* ]] || [[ "$first_part" == *":"* ]]
            then
                # Source already has a registry prefix (e.g., "us-central1-docker.pkg.dev/.../library/ubuntu:24.04")
                skopeo copy --override-os linux --override-arch amd64 docker://$src docker://$2
            else
                # First part is just a namespace (e.g., "envoyproxy/envoy:v1.33.0"), prepend docker.io/
                skopeo copy --override-os linux --override-arch amd64 docker://docker.io/$src docker://$2
            fi
        else
            # Bare image name (e.g., "ubuntu:24.04"), prepend docker.io/
            skopeo copy --override-os linux --override-arch amd64 docker://docker.io/$src docker://$2
        fi
    }
else
    echo Could not find skopeo, falling back to docker which will be slower.
    copy_image() {
        docker pull $1
        docker tag $1 $2
        docker push $2
    }
fi
