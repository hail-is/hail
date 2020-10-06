FROM {{ hail_ubuntu_image.image }}
RUN apt-get update && apt-get install -y socat
CMD ["socat", "TCP4-LISTEN:5000,fork", "EXEC:cat"]
