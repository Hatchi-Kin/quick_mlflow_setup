# quick_mlflow_setup

A simple template to quickly set up in a docker-compose environment:
 - A FastAPI app
 - A MLFlow Tracking Server
 - A MinIO Object Storage 
 - A PostgreSQL Database

## Requirements

- Docker
- Docker Compose


## Build and start the services

```bash
docker compose up -d --build
```

## first time setup
Need to go to the MiniO UI and create a  access key and secret key. 
Then, make a copy of the `.env.example` file and rename it to `.env`.
Update the `.env` file with the access key and secret key.

then restart the services
```bash
docker compose down
docker compose up -d --build
```

## Access the services
- MLFlow Tracking Server: [http://localhost:5001](http://localhost:5001)
- MinIO Console UI: [http://localhost:9001](http://localhost:9001)