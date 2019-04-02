FROM gcr.io/hail-vdc/base

COPY requirements.txt /notebook/
COPY notebook /notebook/
COPY notebook-worker-images /notebook/
COPY hailjwt /notebook/hailjwt/

WORKDIR /notebook
RUN pip3 install -r /notebook/requirements.txt

EXPOSE 5000

ENTRYPOINT ["python3", "notebook.py"]
