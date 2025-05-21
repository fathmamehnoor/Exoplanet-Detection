# Use official Python image
FROM python:3.10

# Set working directory inside the container
WORKDIR /app

# Copy all files to the container's working directory
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Command to run the Prefect flow
CMD ["python", "-m", "prefect_flows.flows"]
