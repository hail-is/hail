FROM base

COPY hailjwt/ /hailjwt/
RUN cd /hailjwt && python3 setup.py install

COPY setup.py /batch/
COPY MANIFEST.in /batch/
COPY batch /batch/batch/
RUN pip3 install --no-cache-dir /batch

EXPOSE 5000

CMD ["python3", "-c", "import batch.server; batch.server.serve()"]
