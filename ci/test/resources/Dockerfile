ARG BASE_IMAGE={{ ci_utils_image.image }}
FROM $BASE_IMAGE

COPY ci/test/resources/hello.py /hello/

EXPOSE 5000

CMD ["python3", "/hello/hello.py"]
