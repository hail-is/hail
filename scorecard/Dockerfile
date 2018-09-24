FROM continuumio/miniconda
MAINTAINER Hail Team <hail@broadinstitute.org>

COPY environment.yml .
RUN conda env create scorecard -f environment.yml && \
    rm -f environment.yml && \
    rm -rf /home/root/.conda/pkgs/*

COPY scorecard /scorecard

EXPOSE 5000

CMD ["bash", "-c", "source activate scorecard; python /scorecard/scorecard.py"]
