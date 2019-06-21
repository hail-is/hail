FROM {{ base_image.image }}

COPY notebook2/notebook/ /notebook/

COPY gear/setup.py /gear/
COPY gear/gear/ /gear/gear/
RUN python3 -m pip install --no-cache-dir /gear \
  && rm -rf /gear

WORKDIR /notebook

EXPOSE 5000

ENTRYPOINT ["python3", "notebook.py"]
