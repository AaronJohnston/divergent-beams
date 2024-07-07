source activate pytorch && python -m gunicorn.app.wsgiapp \
    --access-logfile - \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 1 \
    --bind :8000 \
    --log-level DEBUG \
    main:app