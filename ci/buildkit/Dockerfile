FROM {{ global.docker_prefix }}/quay.io/buildah/stable:v1.30.0
RUN yum install -y python3 python3-pip jq && pip3 install jinja2
COPY --chown=user:user jinja2_render.py /home/user/jinja2_render.py
COPY --chown=user:user buildkit/convert-cloud-credentials-to-docker-auth-config /home/user/convert-cloud-credentials-to-docker-auth-config
