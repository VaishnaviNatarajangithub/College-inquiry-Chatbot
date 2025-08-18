# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all project files into container
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Expose the Flask port
EXPOSE 5000

# Run your chatbot (change train.py to run.py if needed)
CMD ["python", "run.py"]
