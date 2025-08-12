FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
EXPOSE 8000
CMD ["streamlit", "run", "resume_analyzer_app.py", "--server.port=8000", "--server.address=0.0.0.0"]
