FROM python:3.7-slim-stretch

RUN pip3 install aiohttp aiohttp_jinja2

COPY benchmark-service/benchmark /benchmark-service/benchmark/

COPY benchmark-service/setup.py benchmark-service/MANIFEST.in  /benchmark-service/

RUN pip3 install --no-cache-dir /benchmark-service && \
    rm -rf /benchmark-service

EXPOSE 5000

ENTRYPOINT ["python3"]

CMD ["-m", "benchmark.benchmark"]
