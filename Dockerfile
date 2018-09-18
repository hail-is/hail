FROM continuumio/miniconda
MAINTAINER Hail Team <hail@broadinstitute.org>

COPY environment.yml .
COPY scorecard /scorecard
RUN conda env create scorecard -f environment.yml && \
    rm -f environment.yml && \
    rm -rf /home/root/.conda/pkgs/*

EXPOSE 5000

CMD ["bash", "-c", "source activate scorecoard; /scorecard/scorecard.py"]
