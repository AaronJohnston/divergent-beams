source activate pytorch && python -m gunicorn.app.wsgiapp \
    --access-logfile - \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 1 \
    --bind unix:/home/ec2-user/web/run/gunicorn.sock \
    --log-level DEBUG \
    main:app