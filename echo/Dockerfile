FROM {{ hail_ubuntu_image.image }}
RUN hail-apt-get-install socat
CMD ["socat", "TCP4-LISTEN:5000,fork", "EXEC:cat"]
