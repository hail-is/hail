FROM {{ hail_run_image.image }}

RUN useradd -m hail
USER hail
WORKDIR /home/hail
COPY wheel-container.tar .
RUN tar -xvf wheel-container.tar && \
    python3 -m pip install -U hail-*-py3-none-any.whl

ENTRYPOINT []

COPY test_shuffler.py .
