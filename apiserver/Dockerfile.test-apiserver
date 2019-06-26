FROM {{ hail_base_image.image }}

COPY hail/python/setup-hailtop.py /hailtop/setup.py
COPY hail/python/hailtop /hailtop/hailtop/
RUN python3 -m pip install --no-cache-dir /hailtop \
  && rm -rf /hailtop

# FIXME buildImage has the context, we can just COPY these
COPY test.tar.gz .
RUN tar xzvf test.tar.gz && \
  rm -f test.tar.gz

COPY resources.tar.gz .
RUN tar xzvf resources.tar.gz && \
  rm -f resources.tar.gz

COPY data.tar.gz .
RUN tar xzvf data.tar.gz && \
  rm -f data.tar.gz
