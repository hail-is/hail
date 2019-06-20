FROM {{ base_image.image }}

COPY gear/setup.py /gear/
COPY gear/gear/ /gear/gear/
RUN python3 -m pip install --no-cache-dir /gear \
  && rm -rf /gear

WORKDIR /notebook
RUN pip3 install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python3", "notebook.py"]
