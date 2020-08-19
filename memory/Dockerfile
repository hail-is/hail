FROM {{ service_base_image.image }}

RUN wget http://download.redis.io/releases/redis-6.0.6.tar.gz && \
    tar xzf redis-6.0.6.tar.gz && \
    cd redis-6.0.6 && \ 
    make
RUN pip3 install aioredis

COPY memory/setup.py /memory/
COPY memory/redis.conf /redis/
COPY memory/memory /memory/memory/
RUN pip3 install --no-cache-dir /memory && \
  rm -rf /memory

EXPOSE 5000
