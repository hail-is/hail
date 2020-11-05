FROM gcr.io/{{ global.project }}/ubuntu:18.04
RUN apt-get update && apt-get install -y socat
CMD ["socat", "TCP4-LISTEN:5000,fork", "EXEC:cat"]
