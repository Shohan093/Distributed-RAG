#!/usr/bin/env bash
# Installs Redis from Debian's apt repo (simpler than Kafka -- no KRaft-style
# manual config needed) and binds it to the VM's internal IP only, never
# 0.0.0.0 -- Redis has no auth by default, so it must never be reachable
# from outside the VPC. This is what backs driftdrag.ledger.drift_ledger in
# the GCP deployment, same Redis protocol as the local/docker-compose paths.
set -euo pipefail
exec > >(tee /var/log/driftdrag-redis-setup.log) 2>&1

INTERNAL_IP="$(curl -s -H 'Metadata-Flavor: Google' 'http://169.254.169.254/computeMetadata/v1/instance/network-interfaces/0/ip')"

apt-get update -y
apt-get install -y redis-server

sed -i "s/^bind .*/bind 127.0.0.1 ${INTERNAL_IP}/" /etc/redis/redis.conf
sed -i "s/^# maxmemory .*/maxmemory 512mb/" /etc/redis/redis.conf
sed -i "s/^# maxmemory-policy .*/maxmemory-policy allkeys-lru/" /etc/redis/redis.conf
sed -i "s/^appendonly no/appendonly yes/" /etc/redis/redis.conf

systemctl enable redis-server
systemctl restart redis-server

echo "Applied bind config:"
grep '^bind' /etc/redis/redis.conf
echo "Redis listening on ${INTERNAL_IP}:6379 (VPC-internal only). If the bind line above"
echo "doesn't show ${INTERNAL_IP}, edit /etc/redis/redis.conf manually and 'systemctl restart redis-server'."
