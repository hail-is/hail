FROM {{ service_base_image.image }}

COPY benchmark-service/benchmark /benchmark-service/benchmark/

COPY benchmark-service/setup.py benchmark-service/MANIFEST.in  /benchmark-service/

RUN python3 -m pip install --use-feature=2020-resolver --upgrade --no-cache-dir /benchmark-service && \
    rm -rf /benchmark-service && \
    python3 -m pip check

RUN pip3 install plotly

RUN pip3 install pandas

RUN pip3 install scipy

RUN pip3 install numpy

EXPOSE 5000

CMD ["python3", "-m", "benchmark"]
