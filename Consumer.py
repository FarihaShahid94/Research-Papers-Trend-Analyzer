from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StringType, ArrayType
from collections import defaultdict, Counter
import time
import logging
import pandas as pd
from prophet import Prophet
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import uvicorn
import threading
from datetime import datetime

from pyspark.ml.feature import CountVectorizer, IDF
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("StandaloneTest")

trending_keywords_by_category = defaultdict(Counter)
papers_over_time = Counter()

tfidf_by_category = defaultdict(dict)  
tfidf_agg_stats = defaultdict(lambda: {"sum_tfidf": 0.0, "count": 0})

cool_words = set([
    "transformer", "cnn", "rnn", "lstm", "gan", "bert", "gpt", "diffusion", "convolutional",
    "reinforcement", "transfer", "meta-learning", "graph", "attention", "autonomous", "quantum",
    "edge", "federated", "blockchain", "bayesian", "optimization", "swarm", "multi-agent",
    "zero-shot", "few-shot", "semi-supervised", "unsupervised", "neural", "bioinformatics",
    "medical", "robotics", "vision", "nlp", "speech", "detection", "recommendation", "anomaly",
    "synthesis", "mri", "privacy", "secure", "adversarial", "explainable", "interpretable",
    "simulation", "clustering", "bigdata", "data", "scalable", "high-performance", "distributed",
    "kernel", "svm", "pytorch", "tensorflow", "cuda", "gpu", "cloud", "serverless", "hpc",
    "kubernetes", "container", "api", "compiler", "type", "functional", "logic", "formal",
    "theorem", "proof", "approximation", "satisfiability", "boolean", "automata", "compression",
    "parallel", "concurrent", "scheduling", "network", "routing", "protocol", "wireless", "iot",
    "sensor", "cryptography", "encryption", "block", "hash", "secure", "trust", "auth", "malware",
    "penetration", "attack", "defense", "turing", "complexity", "algorithm", "structure", "graph",
    "tree", "path", "regex", "compiler", "cache", "throughput", "latency",
    "transformer", "cnn", "rnn", "lstm", "gan", "bert", "gpt", "diffusion", "convolutional",
    "reinforcement", "transfer", "meta-learning", "attention", "zero-shot", "few-shot",
    "semi-supervised", "unsupervised", "self-supervised", "neural", "deepfake", "foundation-model",
    "multimodal", "prompt", "finetuning", "pretraining", "autoencoder", "vae", "contrastive",
    "embedding", "hyperparameter", "overfitting", "underfitting", "regularization", "dropout",
    "batchnorm", "resnet", "alexnet", "mobilenet", "efficientnet", "nasnet", "swin", "vit",
    "clip", "dalle", "stable-diffusion", "dreambooth", "controlnet", "sam", "llm", "rlhf",
    "chain-of-thought", "few-shot-learning", "meta-learning", "curriculum-learning", "active-learning",
    "semi-supervised-learning", "self-supervised-learning", "unsupervised-learning", "supervised-learning",
    "online-learning", "offline-learning", "continual-learning", "multi-task-learning", "multi-modal-learning",
    "federated-learning", "distributed-learning", "parallel-learning", "ensemble-learning", "boosting",
    "bagging", "stacking", "decision-tree", "random-forest", "xgboost", "lightgbm", "catboost",
    "svm", "knn", "naive-bayes", "logistic-regression", "linear-regression", "ridge-regression",
    "lasso-regression", "elastic-net", "pca", "tsne", "umap", "k-means", "dbscan", "hierarchical-clustering",
    "agglomerative-clustering", "mean-shift", "spectral-clustering", "gaussian-mixture-model", "hmm",
    "markov-decision-process", "q-learning", "sarsa", "policy-gradient", "actor-critic", "ppo", "ddpg",
    "td3", "sac", "dqn", "double-dqn", "dueling-dqn", "prioritized-experience-replay", "multi-agent-rl",
    "inverse-rl", "imitation-learning", "behavioral-cloning", "apprenticeship-learning", "bayesian-network",
    "bayesian-inference", "monte-carlo", "mcmc", "gibbs-sampling", "metropolis-hastings", "variational-inference",
    "elbo", "kl-divergence", "information-gain", "entropy", "mutual-information", "cross-entropy",
    "hinge-loss", "huber-loss", "mae", "mse", "rmse", "roc-curve", "auc", "precision", "recall",
    "f1-score", "confusion-matrix", "precision-recall-curve", "pr-curve", "log-loss", "brier-score",
    "calibration", "probability-calibration", "isotonic-regression", "platt-scaling", "temperature-scaling",
    "label-smoothing", "data-augmentation", "mixup", "cutmix", "cutout", "random-erasing", "adversarial-training",
    "fgsm", "pgd", "deepfool", "cw-attack", "boundary-attack", "universal-perturbation", "black-box-attack",
    "white-box-attack", "gradient-masking", "obfuscated-gradients", "robustness", "certified-robustness",
    "lipschitz-constant", "spectral-norm", "weight-decay", "early-stopping", "learning-rate-scheduler",
    "cosine-annealing", "step-decay", "exponential-decay", "one-cycle-policy", "warmup", "gradient-clipping",
    "gradient-accumulation", "mixed-precision-training", "fp16", "bf16", "quantization", "pruning",
    "knowledge-distillation", "teacher-student", "model-compression", "model-parallelism", "data-parallelism",
    "pipeline-parallelism", "tensor-parallelism", "sharding", "zero", "deepspeed", "megatron", "fairscale",
    "horovod", "nccl", "gloo", "apex", "amp", "onnx", "tensorrt", "openvino", "mlir", "xla", "jax",
    "flax", "haiku", "optax", "elegy", "keras", "pytorch", "tensorflow", "mxnet", "chainer", "cntk",
    "theano", "caffe", "caffe2", "torch", "torchvision", "torchaudio", "torchtext", "huggingface",
    "transformers", "datasets", "tokenizers", "accelerate", "diffusers", "peft", "trl", "sentence-transformers",
    "fastai", "fastcore", "fastdownload", "fastprogress", "lightning", "pytorch-lightning", "ignite",
    "skorch", "sklearn", "scikit-learn", "xgboost", "lightgbm", "catboost", "mlflow", "dvc", "wandb",
    "tensorboard", "comet", "neptune", "clearml", "optuna", "ray", "ray-tune", "hyperopt", "ax", "bohb",
    "spearmint", "smac", "nevergrad", "bayesopt", "dragonfly", "skopt", "hyperband", "asha", "population-based-training",
    "evolutionary-algorithms", "genetic-algorithms", "neuroevolution", "cma-es", "nsga-ii", "multi-objective-optimization",
    "pareto-front", "grid-search", "random-search", "bayesian-optimization", "hyperparameter-tuning",
    "model-selection", "cross-validation", "k-fold", "stratified-k-fold", "leave-one-out", "bootstrap",
    "jackknife", "resampling", "ensemble-methods", "bagging", "boosting", "stacking", "blending",
    "voting-classifier", "hard-voting", "soft-voting", "gradient-boosting", "adaboost", "gbdt", "xgboost",
    "lightgbm", "catboost", "hist-gradient-boosting", "random-forest", "extra-trees", "isolation-forest",
    "one-class-svm", "local-outlier-factor", "elliptic-envelope", "autoencoder", "variational-autoencoder",
    "gan", "dcgan", "wgan", "wgan-gp", "lsgan", "cgan", "pix2pix", "cycle-gan", "style-gan", "style-gan2",
    "biggan", "progan", "sngan", "esrgan", "srgan", "deepfake", "faceapp", "faceswap", "deepfacelab",
    "first-order-motion-model", "talking-head", "neural-style-transfer", "fast-style-transfer",
    "neural-doodle", "neural-painting", "artbreeder", "runwayml", "thispersondoesnotexist",
    "thiscatdoesnotexist", "thiswaifudoesnotexist", "thisrentaldoesnotexist", "thisresume",
    "thisstartup", "thisxdoesnotexist", "deepdream", "deepart", "prisma", "ostagram", "neural-network",
    "deep-learning", "machine-learning", "artificial-intelligence", "ai", "ml", "dl", "nlp", "cv",
    "rl", "gan", "vae", "ae", "cnn", "rnn", "lstm", "gru", "transformer", "attention", "self-attention",
    "multi-head-attention", "positional-encoding", "encoder-decoder", "seq2seq", "beam-search",
    "greedy-decoding", "top-k-sampling", "top-p-sampling", "nucleus-sampling", "temperature-sampling",
    "repetition-penalty", "length-penalty", "coverage-penalty", "diverse-beam-search", "no-repeat-ngram",
    "min-length", "max-length", "early-stopping", "stop-sequence", "stop-token", "eos-token", "bos-token",
    "pad-token", "unk-token", "mask-token", "cls-token", "sep-token", "tokenization", "byte-pair-encoding",
    "wordpiece", "sentencepiece", "unigram", "subword", "tokenizer", "vocab", "vocabulary", "embedding",
    "word-embedding", "contextual-embedding", "static-embedding", "glove", "word2vec", "fasttext",
    "elmo", "bert", "roberta", "albert", "distilbert", "electra", "xlnet", "flaubert", "camembert",
    "bart", "mbart", "t5", "mt5", "pegasus", "prophetnet", "longformer", "reformer", "linformer",
    "bigbird", "performer", "synthesizer", "gpt", "gpt2", "gpt3", "gpt-neo", "gpt-j", "gpt-neox",
    "gpt4", "chatgpt", "instructgpt", "codex", "copilot", "codegen", "polycoder", "incoder",
    "alphacode", "codeparrot", "codebert", "graphcodebert", "unixcoder", "codet5", "codegpt"
])

cool_word_counts_by_category = defaultdict(Counter)  


stopwords = set([
    "the", "and", "but", "with", "for", "from", "this", "that", "have", "has",
    "are", "was", "were", "will", "can", "just", "into", "about", "more", "some",
    "any", "other", "such", "not", "than", "too", "very", "own", "also", "via",
    "yet", "shall", "then", "upon", "only", "many", "few", "all", "our", "your"
])

spark = SparkSession.builder \
    .appName("ArxivTrendAnalyzerTest") \
    .master("local[*]") \
    .getOrCreate()

spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
spark.sparkContext.setLogLevel("WARN")

schema = StructType() \
    .add("title", StringType()) \
    .add("summary", StringType()) \
    .add("published", StringType()) \
    .add("categories", ArrayType(StringType())) \
    .add("authors", ArrayType(StringType()))

df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "arxiv-papers") \
    .option("startingOffsets", "latest") \
    .load()

df_parsed = df_raw.selectExpr("CAST(value AS STRING) as json_str") \
    .select(from_json(col("json_str"), schema).alias("data")) \
    .filter(col("data").isNotNull()) \
    .select("data.*")

df_exploded = df_parsed.withColumn("category", explode(col("categories")))

text_df = df_exploded.withColumn("text", lower(concat_ws(" ", col("summary"))))
words_df = text_df.withColumn("word", explode(split(col("text"), "\\W+")))
filtered_df = words_df.filter(
    (col("word").rlike("^[a-zA-Z]{3,}$")) & (~col("word").isin(stopwords))
)
filtered_df = filtered_df.withColumn(
    "event_time", to_timestamp(col("published"), "EEE, dd MMM yyyy HH:mm:ss zzz")
)

trend_df = filtered_df \
    .withWatermark("event_time", "15 minutes") \
    .groupBy(
        window(col("event_time"), "10 minutes", "5 minutes"),
        col("category"),
        col("word")
    ).count()

def extract_top_tfidf_words(vocab, tfidf_vector, top_n=10):
    import numpy as np
    arr = tfidf_vector.toArray()
    top_indices = np.argsort(arr)[::-1][:top_n]
    top_words = [(vocab[i], float(arr[i])) for i in top_indices if arr[i] > 0]
    return top_words

def process_trend_batch(df, epoch_id):
    start = time.time()
    df.persist()
    row_count = df.count()
    logger.info(f"Processing trend batch {epoch_id} with {row_count} rows")
    if row_count == 0:
        return

    batch_data = df.collect()
    for row in batch_data:
        category = row['category']
        word = row['word']
        cnt = row['count']
        trending_keywords_by_category[category][word] += cnt
        ts_bucket = row['window'].start.strftime('%Y-%m-%d %H:%M:%S')
        papers_over_time[ts_bucket] += cnt

    if word.lower() in cool_words:
        cool_word_counts_by_category[category][word.lower()] += cnt

    df_cat_words = df.groupBy("category").agg(
        collect_list("word").alias("words")
    )

    if df_cat_words.count() == 0:
        logger.info("No category words to process TF-IDF")
    else:
        cv = CountVectorizer(inputCol="words", outputCol="tf_features", vocabSize=1000, minDF=1.0)
        cv_model = cv.fit(df_cat_words)
        df_vectorized = cv_model.transform(df_cat_words)

        idf = IDF(inputCol="tf_features", outputCol="tfidf_features")
        idf_model = idf.fit(df_vectorized)
        df_tfidf = idf_model.transform(df_vectorized)

        for row in df_tfidf.collect():
            cat = row['category']
            vocab = cv_model.vocabulary
            tfidf_vec = row['tfidf_features']
            top_words = extract_top_tfidf_words(vocab, tfidf_vec, top_n=10)
            tfidf_by_category[cat] = dict(top_words)

            total_score = __builtins__.sum(score for _, score in top_words)
            count_words = len(top_words)
            if count_words > 0:
                tfidf_agg_stats[cat]["sum_tfidf"] += total_score
                tfidf_agg_stats[cat]["count"] += count_words

    logger.info("Top keywords in 'cs.AI': %s", trending_keywords_by_category.get("cs.AI", {}).most_common(5))
    logger.info("Top TF-IDF words in 'cs.AI': %s", tfidf_by_category.get("cs.AI", {}))
    logger.info("Recent paper publish times: %s", dict(list(papers_over_time.items())[-5:]))
    logger.info(f"Trend batch {epoch_id} done in {time.time() - start:.2f}s")

query_trend = trend_df.writeStream \
    .outputMode("complete") \
    .foreachBatch(process_trend_batch) \
    .start()

last_forecast_df = None
forecast_lock = threading.Lock()

def forecast_with_prophet():
    logger.info("In forecasting")
    global last_forecast_df
    if not papers_over_time:
        logger.info("no data for forecasting")
        print(" No data available for forecasting.")
        return
    df_forecast = pd.DataFrame(list(papers_over_time.items()), columns=["ds", "y"])
    df_forecast["ds"] = pd.to_datetime(df_forecast["ds"])
    df_forecast.sort_values("ds", inplace=True)
    model = Prophet()
    model.fit(df_forecast)
    future = model.make_future_dataframe(periods=5, freq="10min")
    forecast = model.predict(future)
    with forecast_lock:
        last_forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    print("\n Forecast updated")

app = FastAPI()
 
origins = [
    "http://localhost:3000",  
]
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/keywords/{category}")
def get_top_keywords(category: str, top: int = 10):
    keywords = trending_keywords_by_category.get(category)
    if not keywords:
        return JSONResponse(status_code=404, content={"message": "Category not found or no data"})
    top_keywords = keywords.most_common(top)
    return {"category": category, "top_keywords": top_keywords}

@app.get("/categories")
def get_categories():
    cats = list(trending_keywords_by_category.keys())
    return {"categories": cats}

@app.get("/keywords/{category}/word/{word}")
def get_keyword_frequency(category: str, word: str):
    keywords = trending_keywords_by_category.get(category)
    if not keywords:
        return JSONResponse(status_code=404, content={"message": "Category not found or no data"})
    freq = keywords.get(word.lower(), 0)
    return {"category": category, "word": word.lower(), "frequency": freq}

@app.get("/papers_over_time")
def get_papers_over_time():
    data = sorted(papers_over_time.items())
    return {"papers_over_time": data}

@app.get("/papers_over_time/{start}/{end}")
def get_papers_over_time_range(start: str, end: str):
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return JSONResponse(status_code=400, content={"message": "Invalid datetime format. Use YYYY-MM-DD HH:MM:SS"})
    filtered = [(ts, cnt) for ts, cnt in papers_over_time.items()
                if start <= ts <= end]
    filtered.sort()
    return {"papers_over_time_range": filtered, "start": start, "end": end}

@app.get("/forecast")
def get_forecast():
    with forecast_lock:
        if last_forecast_df is None:
            return JSONResponse(status_code=404, content={"message": "No forecast available yet"})
        forecast_json = last_forecast_df.tail(5).to_dict(orient="records")
    return {"forecast": forecast_json}

@app.get("/forecast/summary")
def get_forecast_summary():
    with forecast_lock:
        if last_forecast_df is None:
            return JSONResponse(status_code=404, content={"message": "No forecast available yet"})
        avg_pred = last_forecast_df["yhat"].tail(5).mean()
    return {"average_predicted_papers_next_5_intervals": avg_pred}

@app.get("/forecast/interval")
def get_forecast_interval(index: int = Query(..., ge=0, le=4)):
    with forecast_lock:
        if last_forecast_df is None:
            return JSONResponse(status_code=404, content={"message": "No forecast available yet"})
        forecast_rows = last_forecast_df.tail(5).reset_index(drop=True)
        if index >= len(forecast_rows):
            return JSONResponse(status_code=400, content={"message": "Index out of range"})
        row = forecast_rows.iloc[index]
        return {
            "ds": row["ds"].strftime("%Y-%m-%d %H:%M:%S"),
            "yhat": row["yhat"],
            "yhat_lower": row["yhat_lower"],
            "yhat_upper": row["yhat_upper"],
        }

@app.get("/trending_words")
def get_overall_trending_words(top: int = Query(10, ge=1, le=100)):
    combined = Counter()
    for cat_counter in trending_keywords_by_category.values():
        combined.update(cat_counter)
    top_words = combined.most_common(top)
    return {"top_trending_words": top_words}

@app.get("/stats")
def get_overall_stats():
    total_categories = len(trending_keywords_by_category)
    unique_keywords = len(set(k for counter in trending_keywords_by_category.values() for k in counter.keys()))
    total_papers = __builtins__.sum(papers_over_time.values())
    tfidf_summary = {cat: (stats["sum_tfidf"] / stats["count"] if stats["count"] > 0 else 0.0)
                     for cat, stats in tfidf_agg_stats.items()}
    return {
        "total_categories": total_categories,
        "total_unique_keywords": unique_keywords,
        "total_papers_counted": total_papers,
        "average_tfidf_per_category": tfidf_summary
    }


@app.get("/tfidf/{category}")
def get_top_tfidf_words(category: str, top: int = 10):
    words_scores = tfidf_by_category.get(category)
    if not words_scores:
        return JSONResponse(status_code=404, content={"message": "Category not found or no TF-IDF data"})
    top_words = sorted(words_scores.items(), key=lambda x: x[1], reverse=True)[:top]
    return {"category": category, "top_tfidf_words": top_words}

@app.get("/tfidf")
def get_tfidf_summary():
    summary = {}
    for cat, stats in tfidf_agg_stats.items():
        avg_tfidf = stats["sum_tfidf"] / stats["count"] if stats["count"] > 0 else 0.0
        summary[cat] = avg_tfidf
    return {"average_tfidf_per_category": summary}

@app.get("/aggregation/top_keywords")
def get_top_keywords_aggregated(top: int = 10):
    combined = Counter()
    for cat_counter in trending_keywords_by_category.values():
        combined.update(cat_counter)
    top_words = combined.most_common(top)
    return {"top_keywords_overall": top_words}

@app.get("/aggregation/total_papers_per_category")
def get_total_papers_per_category():
    total_per_cat = {cat: __builtins__.sum(counter.values()) for cat, counter in trending_keywords_by_category.items()}
    return {"total_papers_per_category": total_per_cat}

@app.get("/coolword-counts/{category}")
def get_cool_word_counts(category: str):
    counts = cool_word_counts_by_category.get(category)
    if not counts:
        return JSONResponse(status_code=404, content={"message": "Category not found or no cool words"})
    
    total = __builtins__.sum(counts.values())
    return {
        "category": category,
        "cool_word_count": total,
        "matched_words": dict(counts)
    }

@app.get("/coolword-counts")
def get_all_cool_word_counts():
    return {
        cat: {
            "cool_word_count": __builtins__.sum(counts.values()),
            "matched_words": dict(counts)
        } for cat, counts in cool_word_counts_by_category.items()
    }

@app.get("/all-coolword")
def get_all_cool_word():
    return{
        "coolwords" : cool_words
    }

def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=run_api, daemon=True).start()

logger.info("Forecasting called")

def schedule_forecast():
    forecast_with_prophet()
    threading.Timer(600, schedule_forecast).start()

schedule_forecast()

query_trend.awaitTermination()
