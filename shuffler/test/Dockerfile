FROM {{ hail_run_image.image }}

RUN useradd -m hail
USER hail
WORKDIR /home/hail
RUN python3 -m pip install -U /hail.whl

ENTRYPOINT []

COPY test_shuffler.py .
