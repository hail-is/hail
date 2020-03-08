FROM {{ dbuf_image.image }}

RUN useradd -m hail
USER hail
WORKDIR /home/hail
RUN python3 -m pip install -U numpy

ENTRYPOINT []

COPY scale_test.py .
