FROM {{ base_image.image }}

COPY hail/python/setup-hailtop.py /hailtop/setup.py
COPY hail/python/hailtop /hailtop/hailtop/
RUN python3 -m pip install --no-cache-dir /hailtop \
  && rm -rf /hailtop

COPY router-resolver/router_resolver /router_resolver

EXPOSE 5000

CMD ["python3", "/router_resolver/router_resolver.py"]
