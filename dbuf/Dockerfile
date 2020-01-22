FROM {{ service_base_image.image }}

COPY setup.py dbuf/
COPY dbuf dbuf/dbuf
RUN python3 -m pip install ./dbuf

ENTRYPOINT ["python3", "-m", "dbuf"]
CMD []
