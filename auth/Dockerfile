FROM {{ service_base_image.image }}

RUN pip3 install --upgrade \
      google-auth-oauthlib==0.4.1 \
      google-auth==1.20.0

COPY auth/setup.py auth/MANIFEST.in /auth/
COPY auth/auth /auth/auth/
RUN pip3 install --use-feature=2020-resolver --upgrade --no-cache-dir /auth && \
  rm -rf /auth && \
  python3 -m pip check

EXPOSE 5000
