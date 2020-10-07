FROM {{ service_base_image.image }}

RUN python3 -m pip install  --use-feature=2020-resolver --upgrade --no-cache-dir \
      google-auth-oauthlib==0.4.1 \
      google-auth==1.21.1 && \
    python3 -m pip check

COPY auth/setup.py auth/MANIFEST.in /auth/
COPY auth/auth /auth/auth/
RUN python3 -m pip install --use-feature=2020-resolver --upgrade --no-cache-dir /auth && \
  rm -rf /auth && \
  python3 -m pip check

EXPOSE 5000
