FROM python:3.6-slim

COPY setup.py setup.cfg dbuf/
COPY dbuf dbuf/dbuf
RUN python3 -m pip install ./dbuf

ENTRYPOINT ["python3", "-m", "dbuf.dbuf"]
CMD ["1"]
