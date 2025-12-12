# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Install system dependencies needed for python packages with C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY ./src ./src

# Define environment variable for the port, with a default value
ENV PORT 8000

# Make port $PORT available to the world outside this container
EXPOSE $PORT

# Define environment variable to allow imports from src
ENV PYTHONPATH=/app

# Command to run the application.
# uvicorn will automatically use the PORT environment variable.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0"]
