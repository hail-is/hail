FROM {{ service_base_image.image }}

COPY benchmark-service/benchmark /benchmark-service/benchmark/

COPY benchmark-service/setup.py benchmark-service/MANIFEST.in  /benchmark-service/

COPY benchmark-service/resources/0.2.45-ac6815ee857c-master.json /

RUN pip3 install --no-cache-dir /benchmark-service && \
    rm -rf /benchmark-service

EXPOSE 5000

CMD ["python3", "-m", "benchmark.benchmark"]