FROM {{ service_base_image.image }}

RUN useradd -m dbuf
USER dbuf
WORKDIR /home/dbuf

COPY setup.py setup.cfg dbuf/
COPY dbuf dbuf/dbuf
RUN python3 -m pip install ./dbuf

ENTRYPOINT ["python3", "-m", "dbuf"]
CMD ["1"]
