FROM moby/buildkit:v0.8.3-rootless
USER root
RUN apk add python3 py-pip && pip3 install jinja2
USER user
COPY --chown=user:user jinja2_render.py /home/user/jinja2_render.py
COPY --chown=user:user buildkit/convert-google-application-credentials-to-docker-auth-config /home/user/convert-google-application-credentials-to-docker-auth-config
