# High-Dimensional Image Processing Microservice

This project is a FastAPI-based microservice for processing high-dimensional scientific images (e.g., 5D microscopy or hyperspectral images). It provides functionality for uploading, slicing, analyzing, and retrieving metadata from multi-dimensional TIFF images.

---

## **Features**
- Upload and process 5D TIFF images.
- Extract specific slices (X, Y, Z, Time, Channel).
- Perform Principal Component Analysis (PCA) for dimensionality reduction.
- Calculate image statistics (mean, standard deviation, min, max).
- Asynchronous processing using Celery and Redis.
- Store metadata and analysis results in a PostgreSQL database.
- Dockerized for easy deployment.

---

## **Technologies Used**
- **Python 3.9**
- **FastAPI**: For building the API.
- **Celery**: For asynchronous task processing.
- **Redis**: As a message broker for Celery.
- **PostgreSQL**: For storing image metadata.
- **Docker**: For containerization.
- **Scikit-learn**: For PCA and segmentation.
- **Scikit-image**: For image processing.
- **Tifffile**: For reading multi-dimensional TIFF images.

---

## **Setup and Installation**

### **Prerequisites**
- Docker and Docker Compose installed on your machine.

### **Steps to Run the Application**
1. Clone the repository:
   ```bash
   git@github.com:Raiyan-sharif/High-Dimensional-Image-Processing.git
2. Build the repository:
   ```bash
   docker-compose up --build
3. Access the API:

The API will be available at http://localhost:8000.

Interactive API documentation (Swagger UI) is available at http://localhost:8000/docs.