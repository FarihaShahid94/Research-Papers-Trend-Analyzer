
# Real-Time Multi-Domain Academic Paper Trend Analyzer

This project is a real-time streaming pipeline that analyzes academic paper trends using data from [arXiv](https://arxiv.org/). The pipeline uses **Apache Kafka**, **Apache Spark**, **FastAPI**, and **Next.js** to ingest, process, and visualize metadata from newly published research papers across multiple domains.

## 🚀 Tech Stack

- **Kafka** – Real-time data ingestion pipeline
- **Spark Streaming** – Batch processing and trend analysis (TF-IDF, keyword extraction)
- **FastAPI** – Backend API to serve Spark results
- **Next.js** – Dashboard to visualize analytics
- **arXiv API** – Source of live research paper metadata

## 🛠️ Setup Instructions

### Start Zookeeper

```
bin\windows\zookeeper-server-start.bat config\zookeeper.properties
```

### Start Kafka Server

```
bin\windows\kafka-server-start.bat config\server.properties
```

### Kafka Topic Operations

**List Topics:**
```
bin\windows\kafka-topics.bat --list --bootstrap-server localhost:9092
```

**Describe Topic:**
```
bin\windows\kafka-topics.bat --describe --topic arxiv-papers --bootstrap-server localhost:9092
```

---

## 🧠 Hadoop Setup (If required for HDFS)

```
jps 
hdfs namenode -format
start-all.cmd
stop-all.cmd
```

---

## 🔥 PySpark Usage

Navigate to your PySpark `bin` folder and run:

```
pyspark
```

To submit Spark jobs:

```
spark-submit Producer_apitokafka.py
spark-submit Producer_apitoxml.py
spark-submit Producer_xmltokafka.py
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 Consumer.py
```

📌 **Note:** Always include the correct Kafka JAR packages in `spark-submit` to avoid dependency errors.

---

## 📁 Project Structure

```
arxiv-trend-analyzer/
├── kafka_producers/
├── spark_processor/
├── backend_fastapi/
├── frontend_nextjs/
├── docs/
└── data/
```

## 📌 Reminder

- Ensure Zookeeper and Kafka are running before starting any producers or consumers.
- Use Producer 3 for consistent batch sizes (fixed window of 10,000) when analyzing historical/local data.
- Producer 1 is ideal for real-time data when arXiv APIs are available.
- Weekend unavailability is handled using local XML storage.

---

## 📷 Dashboard & Visuals

Visualizations are available on the Next.js dashboard, which connects to the FastAPI backend for real-time updates.

---