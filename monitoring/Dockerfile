FROM {{ service_base_image.image }}

COPY monitoring/resources/test_billing_data.json /

COPY monitoring/monitoring /monitoring/monitoring/

COPY monitoring/setup.py monitoring/MANIFEST.in  /monitoring/

RUN pip3 install --no-cache-dir /monitoring && \
    rm -rf /monitoring

EXPOSE 5000
