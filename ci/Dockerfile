FROM {{ base_image.image }}

COPY hail/python/setup-hailtop.py /hailtop/setup.py
COPY hail/python/hailtop /hailtop/hailtop/
RUN python3 -m pip install --no-cache-dir /hailtop \
  && rm -rf /hailtop

COPY ci/requirements.txt .
RUN python3 -m pip install --no-cache-dir -U -r requirements.txt

# FIXME install
COPY ci/ci /ci

EXPOSE 5000

CMD ["python3", "-c", "import ci; ci.run()"]
