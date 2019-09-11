FROM {{ service_base_image.image }}

COPY notebook2/notebook/ /notebook/

EXPOSE 5000

ENTRYPOINT ["python3", "-m", "notebook"]
