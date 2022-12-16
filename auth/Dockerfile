FROM {{ service_base_image.image }}

COPY auth/setup.py auth/MANIFEST.in /auth/
COPY auth/auth /auth/auth/
RUN hail-pip-install /auth && rm -rf /auth

EXPOSE 5000
