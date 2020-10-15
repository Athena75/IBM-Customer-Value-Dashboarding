FROM python:3

ADD requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt
ADD . /app
EXPOSE 8050

CMD ["gunicorn", "-b", "0.0.0.0:8050", "run:server"]