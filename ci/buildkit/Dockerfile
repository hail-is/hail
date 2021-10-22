FROM moby/buildkit:v0.8.3-rootless
USER root
RUN apk add python3 py-pip jq && pip3 install jinja2
USER user
COPY --chown=user:user jinja2_render.py /home/user/jinja2_render.py
COPY --chown=user:user buildkit/convert-cloud-credentials-to-docker-auth-config /home/user/convert-cloud-credentials-to-docker-auth-config
