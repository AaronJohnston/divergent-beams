# Tool Config

This directory holds config files used for the tool running at https://aaronjohnston.me/divergent-beams. They are independent of the tool server itself but may be useful inspiration for others looking to host it.

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
