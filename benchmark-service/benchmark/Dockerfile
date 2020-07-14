FROM python:3.7-slim-stretch

COPY benchmark-service/benchmark /benchmark-service/benchmark/

COPY benchmark-service/setup.py benchmark-service/MANIFEST.in  /benchmark-service/

RUN pip install aiohttp aiohttp_jinja2

RUN pip3 install --no-cache-dir /benchmark-service
 # && \ rm -rf /benchmark-service

EXPOSE 5000

#last cmd automatically runs benchmark module, may just want to distribute image
CMD ["python3", "-m", "benchmark.benchmark"]




