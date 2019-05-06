FROM {{ base_image.image }}

COPY notebook2/requirements.txt /notebook/
COPY notebook2/notebook/ /notebook/
COPY hailjwt/ /notebook/hailjwt/

WORKDIR /notebook
RUN pip3 install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python3", "notebook.py"]
