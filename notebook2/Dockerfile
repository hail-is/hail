FROM {{ base_image.image }}

COPY notebook2/notebook/ /notebook/

COPY hail/python/setup-hailtop.py /hailtop/setup.py
COPY hail/python/hailtop /hailtop/hailtop/
RUN python3 -m pip install --no-cache-dir /hailtop \
  && rm -rf /hailtop

EXPOSE 5000

ENTRYPOINT ["python3", "-m", "notebook"]
