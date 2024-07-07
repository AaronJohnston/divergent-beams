# Tool Config

This directory holds config files used for the tool running at https://aaronjohnston.me/divergent-beams. They are independent of the tool server source code; other installations can use entirely different configs, but these files may be useful inspiration.

The server that powers https://aaronjohnston.me/divergent-beams uses a FastAPI service managed by systemd, running behind a Caddy reverse proxy for automatic SSL support. The frontend bundle is served statically from a different source.

## Setup notes

- Install Caddy
  - On Amazon Linux 2: (from http://www.asheiduk.de/post/install-caddy-on-amazon-linux-2/)

```
sudo yum -y install yum-plugin-copr
sudo yum -y copr enable @caddy/caddy epel-7-$(arch)
sudo yum -y install caddy
```

- Copy Caddyfile and divergent-beams.caddy to `/etc/caddy/`
- Copy gunicorn.service to `/etc/systemd/system/`
- `sudo chmod a+x /etc/systemd/system/gunicorn.service`
- `chmod +x config/start.sh`
- `sudo systemctl daemon-reload`
- `sudo systemctl start gunicorn`
