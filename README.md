# VITid
Identificaiton Server for our VIT models

## Setup

- setup venv as normal. Install all of requirements doc.
- install below package  for cuda support:

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Model

- download from our google drive (due to file size limitations on GH)
- add to folder called MODEL


## Deploying the web server

https://docs.gunicorn.org/en/latest/ 

```bash
sudo lsof -i :80
sudo kill <pid of any listed>

sudo ufw allow 80

gunicorn --bind 0.0.0.0:8000 --worker-class gevent idServer:app
cp ./nginx.conf /etc/nginx
```