# mlflow-setup

This repository contains the code for setting up MLFlow Tracking Server with PostgreSQL as backend and MinIO as artifact store, using docker-compose.



## Build and start the services

in the docker folder:

```bash
docker compose up -d --build
```


- MLFlow Tracking Server: [http://localhost:5000](http://localhost:5000)
- MinIO Console UI: [http://localhost:9001](http://localhost:9001)