# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Install virtualenv
RUN pip install virtualenv

# Create a virtual environment and set it as the Python environment
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt into the container at /app
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Set environment variable (if needed for your specific application)
ENV PYTHON_APP model_train.py

# Expose the port on which the app will run (change the port if needed)
EXPOSE 8000

CMD ["python", "model_train.py"]
