FROM gcr.io/hail-vdc/base
LABEL maintainer="hail-team@broadinstitute.org"

COPY requirements.txt /notebook/
COPY notebook /notebook/
COPY notebook-worker-images /notebook/
COPY hailjwt /notebook/hailjwt/

WORKDIR /notebook
RUN pip3 install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python3", "notebook.py"]
