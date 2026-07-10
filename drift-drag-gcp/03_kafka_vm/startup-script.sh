#!/usr/bin/env bash
# Runs automatically on first boot (passed via --metadata-from-file
# startup-script=... in create_kafka_vm.sh). Installs a single-node Kafka
# broker in KRaft mode (no Zookeeper) -- the same combined-mode setup as
# drift-drag/docker-compose.yml's kafka service, just on a bare VM instead
# of a container.
set -euo pipefail
exec > >(tee /var/log/driftdrag-kafka-setup.log) 2>&1

KAFKA_VERSION="3.8.0"
SCALA_VERSION="2.13"
KAFKA_HOME="/opt/kafka"
LOG_DIR="/var/lib/kafka-logs"
INTERNAL_IP="$(curl -s -H 'Metadata-Flavor: Google' 'http://169.254.169.254/computeMetadata/v1/instance/network-interfaces/0/ip')"

echo "Installing Java..."
apt-get update -y
apt-get install -y openjdk-17-jre-headless

echo "Downloading Kafka ${KAFKA_VERSION}..."
cd /opt
curl -sL -o kafka.tgz "https://archive.apache.org/dist/kafka/${KAFKA_VERSION}/kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz"
tar -xzf kafka.tgz
mv "kafka_${SCALA_VERSION}-${KAFKA_VERSION}" "$KAFKA_HOME"
rm kafka.tgz

echo "Configuring KRaft (combined broker+controller, single node)..."
mkdir -p "$LOG_DIR"
CLUSTER_ID="$("$KAFKA_HOME/bin/kafka-storage.sh" random-uuid)"

cat > "$KAFKA_HOME/config/kraft/server.properties" <<EOF
process.roles=broker,controller
node.id=1
controller.quorum.voters=1@localhost:9093
listeners=PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
advertised.listeners=PLAINTEXT://${INTERNAL_IP}:9092
controller.listener.names=CONTROLLER
listener.security.protocol.map=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
inter.broker.listener.name=PLAINTEXT
log.dirs=${LOG_DIR}
num.partitions=8
offsets.topic.replication.factor=1
transaction.state.log.replication.factor=1
transaction.state.log.min.isr=1
EOF

"$KAFKA_HOME/bin/kafka-storage.sh" format \
  -t "$CLUSTER_ID" \
  -c "$KAFKA_HOME/config/kraft/server.properties"

echo "Creating systemd service..."
cat > /etc/systemd/system/kafka.service <<EOF
[Unit]
Description=Apache Kafka (KRaft, single node)
After=network.target

[Service]
Type=simple
User=root
ExecStart=${KAFKA_HOME}/bin/kafka-server-start.sh ${KAFKA_HOME}/config/kraft/server.properties
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable kafka
systemctl start kafka

echo "Waiting for Kafka to come up..."
for i in $(seq 1 30); do
  if "$KAFKA_HOME/bin/kafka-broker-api-versions.sh" --bootstrap-server localhost:9092 >/dev/null 2>&1; then
    echo "Kafka is up."
    break
  fi
  sleep 2
done

echo "Creating topics (docs-raw, logs-raw)..."
"$KAFKA_HOME/bin/kafka-topics.sh" --create --if-not-exists \
  --bootstrap-server localhost:9092 --topic docs-raw --partitions 8 --replication-factor 1
"$KAFKA_HOME/bin/kafka-topics.sh" --create --if-not-exists \
  --bootstrap-server localhost:9092 --topic logs-raw --partitions 8 --replication-factor 1
"$KAFKA_HOME/bin/kafka-topics.sh" --create --if-not-exists \
  --bootstrap-server localhost:9092 --topic shard-assignments --partitions 8 --replication-factor 1

echo "Kafka setup complete. Internal bootstrap address: ${INTERNAL_IP}:9092"
