# بەکارهێنانی پایتۆن ڤێرژنی 3.10
FROM python:3.10

# دروستکردنی فۆڵدەری ئیشکردن
WORKDIR /app

# کۆپیکردنی هەموو فایلەکان بۆ ناو سێرڤەرەکە
COPY . /app

# دابەزاندنی پاکێجەکان
RUN pip install --no-cache-dir -r requirements.txt

# پێدانی دەسەڵات (بۆ ئەوەی کێشەی سیکیوریتی دروست نەبێت لە Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# کارپێکردنی FastAPI لەسەر پۆرتی 7860 (Hugging Face تەنها ئەم پۆرتە قبوڵ دەکات)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
