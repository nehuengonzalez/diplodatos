FROM jupyter/datascience-notebook:df4a9681f19c

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN rm requirements.txt

WORKDIR $HOME



COPY workspace  work
