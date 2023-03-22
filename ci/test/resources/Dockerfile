FROM {{ ci_utils_image.image }}

COPY ci/test/resources/hello.py /hello/

EXPOSE 5000

CMD ["python3", "/hello/hello.py"]
