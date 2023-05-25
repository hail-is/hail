FROM {{ global.docker_prefix }}/quay.io/buildah/stable:v1.30.0
RUN yum install -y python3 python3-pip jq && pip3 install jinja2
COPY jinja2_render.py /jinja2_render.py
COPY buildkit/convert-cloud-credentials-to-docker-auth-config /convert-cloud-credentials-to-docker-auth-config
