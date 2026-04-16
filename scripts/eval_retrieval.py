#!/usr/bin/env python3
"""Retrieval quality measurement: pure-vector / hybrid-RRF / hybrid+composite / hybrid+cross-encoder.

Builds a 300-doc adversarial corpus with 40 queries, measures Recall@5, P@5, P@1, MRR
across four retrieval modes, and prints the 4×4 table.

Corpus design (adversarial categories):
  - 50 core docs (ground-truth targets, one per query topic)
  - 250 confounders (5 per core doc):
      A: Same entity, different context (Python snake vs Python language)
      B: Paraphrase trap (query words present, wrong topic)
      C: Keyword hijack (exact query terms, off-topic content)
      D: Near-duplicate title (same heading, different body)
      E: Semantic neighbour (same domain, slightly different concept)

Usage:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --model cross-encoder/ms-marco-MiniLM-L-6-v2
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import tempfile
import pathlib

# ---------------------------------------------------------------------------
# Adversarial corpus definition
# ---------------------------------------------------------------------------

# Each entry: (chunk_id, content)
# IDs starting with "c_" are core (ground-truth) docs.
# IDs starting with "x_" are confounders.

CORE_DOCS: list[tuple[str, str]] = [
    # --- Programming Languages ---
    ("c_py_interp",   "Python is an interpreted, high-level, general-purpose programming language created by Guido van Rossum. It uses dynamic typing and garbage collection. Its design philosophy emphasizes code readability with significant indentation. Python supports multiple programming paradigms including structured, object-oriented, and functional programming."),
    ("c_py_async",    "Python's asyncio module provides infrastructure for writing single-threaded concurrent code using coroutines, multiplexing I/O access, running network clients and servers, and other related primitives. The async/await syntax added in Python 3.5 makes asynchronous code readable and maintainable."),
    ("c_go_concur",   "Go achieves concurrency through goroutines and channels. A goroutine is a lightweight thread managed by the Go runtime, far cheaper than OS threads. Channels provide type-safe communication between goroutines, following the philosophy: do not communicate by sharing memory; share memory by communicating."),
    ("c_rust_own",    "Rust's ownership system enforces memory safety at compile time without a garbage collector. Each value has a single owner, and when the owner goes out of scope, the value is dropped. Borrowing rules prevent data races: at any time, either one mutable reference or multiple immutable references to a value may exist."),
    ("c_ts_types",    "TypeScript adds optional static typing to JavaScript. Its structural type system checks shapes at compile time, catching type errors before runtime. TypeScript compiles down to plain JavaScript and supports type inference, generics, decorators, and advanced type operations like mapped and conditional types."),
    # --- Machine Learning ---
    ("c_backprop",    "Backpropagation computes gradients of a neural network's loss function with respect to each weight by applying the chain rule of calculus layer by layer. During the backward pass, gradients flow from the output layer to the input layer. Stochastic gradient descent then updates weights proportionally to these gradients scaled by the learning rate."),
    ("c_transformer", "The transformer architecture, introduced in 'Attention Is All You Need', relies entirely on self-attention mechanisms rather than recurrent layers. Multi-head attention allows the model to jointly attend to information from different representation subspaces. Positional encodings inject sequence order since attention is permutation-invariant."),
    ("c_svm",         "Support vector machines find the hyperplane that maximises the margin between classes in a high-dimensional feature space. The kernel trick maps data to higher dimensions implicitly, enabling SVMs to learn non-linear decision boundaries without explicitly computing the transformation. The RBF and polynomial kernels are commonly used for non-linear problems."),
    ("c_kmeans",      "K-means clustering partitions n observations into k clusters by iteratively assigning each point to its nearest centroid and recomputing centroids as the cluster mean. The algorithm minimises within-cluster sum-of-squared distances. Initialisation with k-means++ selects initial centroids with probability proportional to distance, improving convergence."),
    ("c_xgboost",     "XGBoost implements gradient boosted decision trees with several innovations: approximate greedy algorithm for split finding, weighted quantile sketch for efficient data distribution, sparsity-aware split finding for sparse data, and cache-aware block computation. L1 and L2 regularisation terms in the objective prevent overfitting."),
    # --- Databases ---
    ("c_pg_mvcc",     "PostgreSQL uses multi-version concurrency control (MVCC) to allow readers and writers to operate concurrently without locking each other out. Each transaction sees a snapshot of the database consistent as of the transaction start. Old row versions are kept in heap pages and cleaned by VACUUM, which reclaims dead tuples."),
    ("c_mongo_agg",   "MongoDB's aggregation pipeline processes documents through a sequence of stages: $match filters documents, $group accumulates values, $project reshapes output, and $lookup performs left outer joins across collections. Stages execute in sequence; each stage's output feeds the next. Indexes can be used in early $match stages."),
    ("c_redis_ds",    "Redis is an in-memory data structure store supporting strings, hashes, lists, sets, sorted sets with range queries, bitmaps, hyperloglogs, and geospatial indexes. Its atomic operations, pub/sub messaging, and Lua scripting make it suitable for caching, session management, leaderboards, and real-time analytics."),
    ("c_cassandra",   "Apache Cassandra is a wide-column store designed for high availability and linear scalability across multiple datacentres with no single point of failure. It uses consistent hashing to distribute data, tunable consistency levels per operation, and a masterless peer-to-peer replication model based on the gossip protocol."),
    ("c_elastic",     "Elasticsearch is a distributed, RESTful search and analytics engine built on Apache Lucene. It stores data as JSON documents in an inverted index. Shards distribute data across nodes; replicas provide redundancy. The query DSL supports full-text search, structured queries, aggregations, and geo queries with millisecond latency."),
    # --- Security ---
    ("c_aes",         "AES (Advanced Encryption Standard) is a symmetric block cipher operating on 128-bit blocks with key sizes of 128, 192, or 256 bits. Each round applies SubBytes (S-box substitution), ShiftRows (row permutation), MixColumns (column diffusion), and AddRoundKey (XOR with round key). AES-256 performs 14 rounds and is the gold standard for symmetric encryption."),
    ("c_rsa",         "RSA encryption is based on the mathematical difficulty of factoring the product of two large prime numbers. The public key (n, e) encrypts messages; the private key (n, d) decrypts. Key sizes of 2048 bits or larger are recommended. RSA is primarily used for key exchange and digital signatures rather than bulk encryption."),
    ("c_tls",         "TLS (Transport Layer Security) establishes an encrypted, authenticated channel over TCP. The handshake negotiates cipher suites, verifies server identity via X.509 certificates signed by a trusted CA, and derives session keys using ECDHE key exchange for forward secrecy. TLS 1.3 removes obsolete algorithms and reduces handshake latency to one round trip."),
    ("c_xss",         "Cross-site scripting (XSS) attacks inject malicious scripts into web pages viewed by other users. Stored XSS persists injected content in a database; reflected XSS echoes malicious input in a server response. Content Security Policy (CSP) headers, output encoding, and strict HTML sanitisation prevent XSS by restricting script execution."),
    ("c_sqli",        "SQL injection exploits insufficient input validation to insert malicious SQL statements into queries. Attackers can bypass authentication, dump database tables, or execute administrative operations. Parameterised queries (prepared statements) and ORM frameworks prevent injection by separating SQL logic from user-supplied data."),
    # --- Cloud & DevOps ---
    ("c_k8s_sched",   "The Kubernetes scheduler assigns unscheduled pods to nodes by filtering nodes that meet pod requirements (CPU, memory, affinity rules) and then scoring remaining nodes using priority functions. The node with the highest score receives the pod. Custom schedulers and scheduling plugins extend this framework for specialised workloads."),
    ("c_docker_layer","Docker images consist of read-only layers stored in a union filesystem. Each Dockerfile instruction creates a new layer stacked on the previous. When a container starts, a thin writable layer is added on top. The copy-on-write mechanism shares unchanged layers between containers, minimising disk usage and speeding up image pulls."),
    ("c_terraform",   "Terraform manages infrastructure as code by declaring desired resource states in HCL configuration files. The plan command computes a diff between current and desired state; apply executes only the changes needed. State is stored remotely in backends like S3 or Terraform Cloud, enabling collaboration and drift detection."),
    ("c_prometheus",  "Prometheus collects time-series metrics by scraping HTTP endpoints that expose metrics in the OpenMetrics text format. PromQL enables dimensional queries: rate(), increase(), and histogram_quantile() compute rates and percentiles from counter and histogram metrics. Alertmanager routes alerts based on PromQL threshold rules."),
    ("c_nginx",       "nginx is an event-driven web server and reverse proxy that handles thousands of concurrent connections with low memory footprint by using non-blocking I/O and a single-threaded event loop per worker process. It serves static files directly from disk and proxies dynamic requests to upstream application servers via HTTP, FastCGI, or uWSGI."),
    # --- Networking ---
    ("c_tcp_flow",    "TCP congestion control prevents network collapse by limiting the send rate. Slow start grows the congestion window exponentially until a threshold, then linearly (congestion avoidance). On packet loss detected by triple duplicate ACK, CUBIC halves the window; on timeout, it resets to one MSS. BBR estimates bottleneck bandwidth directly."),
    ("c_dns_resolv",  "DNS resolution is recursive: the stub resolver queries a recursive resolver, which contacts root nameservers, then TLD nameservers, then authoritative nameservers to resolve a domain to an IP address. Responses are cached according to TTL values. DNSSEC adds cryptographic signatures to authenticate DNS responses against tampering."),
    ("c_http2",       "HTTP/2 multiplexes multiple request/response streams over a single TCP connection, eliminating head-of-line blocking between requests. Header compression using HPACK reduces overhead on repeat requests. Server push proactively sends resources the client will need. Binary framing replaces HTTP/1.1's text-based protocol."),
    ("c_bgp",         "BGP (Border Gateway Protocol) is the routing protocol of the internet, exchanging reachability information between autonomous systems. iBGP runs within an AS; eBGP peers between ASes. Path selection uses the BGP decision process: prefer highest local preference, then lowest AS path length, then lowest MED, then eBGP over iBGP."),
    ("c_quic",        "QUIC is a transport protocol built on UDP that provides multiplexed streams, TLS 1.3 encryption, and connection migration. Unlike TCP+TLS, QUIC establishes an encrypted connection in a single round trip (0-RTT on resumption). HTTP/3 runs over QUIC, eliminating the TCP head-of-line blocking that affects HTTP/2."),
    # --- Algorithms & Data Structures ---
    ("c_dijkstra",    "Dijkstra's algorithm finds the shortest path from a source node to all other nodes in a weighted graph with non-negative edge weights. It maintains a priority queue (min-heap) of (distance, node) pairs. At each step it extracts the minimum-distance node and relaxes its neighbours. Time complexity is O((V + E) log V) with a binary heap."),
    ("c_bloom",       "A Bloom filter is a probabilistic data structure that tests set membership in constant time and space, with a tunable false-positive rate and zero false negatives. k independent hash functions map each element to k bits in a bit array. Membership test checks all k positions; any zero means the element is definitely absent."),
    ("c_btree",       "B-trees maintain sorted data in a self-balancing tree with branching factor b, keeping all leaf nodes at the same depth. Each internal node holds up to 2b-1 keys and 2b child pointers. Insertions and deletions rebalance locally via splits and merges. B-trees are the dominant structure for database indexes due to low I/O amplification."),
    ("c_lsm",         "Log-structured merge trees write all mutations to an in-memory buffer (memtable) and periodically flush it to immutable sorted string tables (SSTables) on disk. Compaction merges SSTables to limit read amplification. LSM trees achieve high write throughput at the cost of read amplification, making them ideal for write-heavy workloads like key-value stores."),
    ("c_consistent_hash", "Consistent hashing distributes keys across nodes by mapping both keys and nodes to positions on a virtual ring using a hash function. When a node is added or removed, only keys adjacent to that node are redistributed, minimising data movement. Virtual nodes (vnodes) improve load balance by assigning each physical node multiple ring positions."),
    # --- Web Development ---
    ("c_react_fiber", "React's Fiber reconciler performs incremental rendering by breaking work into units that can be paused, resumed, or discarded. The render phase produces a Fiber tree describing desired UI; the commit phase applies mutations to the real DOM synchronously. Concurrent features like startTransition prioritise urgent updates over deferred background work."),
    ("c_graphql",     "GraphQL is a query language for APIs that allows clients to request exactly the data they need in a single request. Unlike REST, which returns fixed resource shapes, GraphQL schemas define types and fields; clients specify their data requirements using a typed query language. Resolvers map each field to its data source."),
    ("c_cors",        "CORS (Cross-Origin Resource Sharing) is a browser security mechanism that restricts cross-origin HTTP requests. The browser sends a preflight OPTIONS request for non-simple requests; the server responds with Access-Control-Allow-Origin and related headers specifying which origins, methods, and headers are permitted."),
    ("c_ssr",         "Server-side rendering (SSR) generates HTML on the server for each request, sending fully populated markup to the browser. Hydration then attaches React (or Vue/Svelte) event handlers to the static HTML. SSR improves First Contentful Paint and SEO compared to client-side rendering, at the cost of server CPU and per-request latency."),
    ("c_websockets",  "WebSockets provide full-duplex communication over a persistent TCP connection. After an HTTP upgrade handshake, client and server exchange frames in both directions without the overhead of HTTP headers per message. WebSockets are used for real-time applications: collaborative editing, live dashboards, multiplayer games, and push notifications."),
    # --- Operating Systems ---
    ("c_mmap",        "Memory-mapped files (mmap) map a file or device into the process virtual address space. Reads and writes to the mapped region trigger page faults that load data from disk on demand. The OS page cache is shared between processes mapping the same file, avoiding redundant copies. mmap is efficient for large read-mostly files and inter-process communication."),
    ("c_epoll",       "Linux epoll is a scalable I/O event notification API replacing select/poll for monitoring thousands of file descriptors. epoll_ctl registers interest in events (EPOLLIN, EPOLLOUT, EPOLLET); epoll_wait returns only ready file descriptors in O(1), making it suitable for high-performance servers handling thousands of concurrent connections."),
    ("c_cgroups",     "Linux control groups (cgroups) limit and account for resource usage of process groups. The memory cgroup sets hard limits on RAM and swap; cpu cgroup restricts CPU time using CFS bandwidth control; blkio cgroup throttles block device I/O. Kubernetes uses cgroups via the container runtime to enforce pod resource limits."),
    ("c_virtual_mem", "Virtual memory abstracts physical RAM using page tables that map virtual to physical addresses. The MMU translates addresses at runtime; TLBs cache recent translations. On a page fault, the OS loads the page from swap or disk. Demand paging defers allocation until access; copy-on-write defers copying until mutation."),
    ("c_ebpf",        "eBPF allows safe execution of sandboxed programs in the Linux kernel without modifying kernel source or loading modules. The verifier statically checks programs for safety before JIT compilation. eBPF programs attach to tracepoints, kprobes, network hooks, or LSM hooks, enabling dynamic tracing, performance analysis, and network filtering without rebooting."),
    # --- Data Engineering ---
    ("c_spark_dag",   "Apache Spark represents computations as a directed acyclic graph (DAG) of RDD transformations. The DAG scheduler splits the DAG at shuffle boundaries into stages; the task scheduler assigns tasks to executors. Lazy evaluation defers computation until an action (collect, write) is called, enabling the optimizer to combine and eliminate stages."),
    ("c_kafka_part",  "Apache Kafka partitions topics across brokers to achieve parallelism and scalability. Producers write to partitions based on a key hash or round-robin; consumers in a consumer group each own exclusive partitions. Replication across brokers ensures durability: the leader handles reads/writes while followers replicate and take over on leader failure."),
    ("c_parquet",     "Parquet stores data in a columnar format: values for each column are stored contiguously, enabling vectorised scans that read only the columns a query needs. Dictionary encoding compresses high-cardinality columns; run-length encoding handles repeated values. Row groups and column chunk statistics enable predicate pushdown, skipping irrelevant data."),
    ("c_dbt",         "dbt (data build tool) transforms raw data in a warehouse by compiling SQL SELECT statements into tables or views. Models reference other models using the ref() function, building a DAG of dependencies. dbt test validates data quality; dbt docs generates a data catalog. Column-level lineage is tracked for impact analysis."),
    ("c_iceberg",     "Apache Iceberg is an open table format for large analytic datasets. It stores table state as a tree of metadata files: a current metadata pointer references a snapshot, which references manifest lists, which reference manifest files listing data files with statistics. Schema evolution, hidden partitioning, and time-travel queries are built in."),
]

# ---------------------------------------------------------------------------
# Confounder pool — 5 per core doc (Type A–E)
# ---------------------------------------------------------------------------

CONFOUNDERS: list[tuple[str, str]] = [
    # Python confounders
    ("x_py_ruby",       "Ruby is an interpreted, high-level, object-oriented programming language known for its clean, readable syntax and dynamic typing. Like Python, Ruby uses garbage collection and supports multiple programming paradigms. The Rails framework made Ruby popular for web development in the early 2000s."),
    ("x_py_perl",       "Perl is an interpreted scripting language with strong support for text processing and regular expressions. Its dynamic typing, flexible syntax, and CPAN module repository made it the dominant language for CGI web scripts and system administration before Python overtook it in popularity."),
    ("x_py_snake",      "Python regius, the ball python, is a non-venomous constrictor native to sub-Saharan Africa. It is one of the most popular pet snake species due to its docile temperament and manageable size. Ball pythons curl into a ball when threatened, which gives them their common name."),
    ("x_py_async2",     "Node.js runs JavaScript on the server using an event-driven, non-blocking I/O model. Like Python's asyncio, it handles concurrent connections without multithreading. The event loop processes callbacks from an event queue, making Node.js efficient for I/O-bound tasks like API servers and real-time applications."),
    ("x_py_indent",     "Haskell is a statically typed functional programming language with lazy evaluation and a strong type system. While unrelated to Python's indentation-based syntax, Haskell also enforces a consistent code layout where whitespace is syntactically significant in do-notation and let expressions."),
    # Python async confounders
    ("x_async_js",      "JavaScript's event loop handles asynchronous operations using a single thread. Promises and async/await syntax, added in ES2015 and ES2017 respectively, make asynchronous code readable. The microtask queue processes resolved promises before the next macrotask, enabling predictable execution order."),
    ("x_async_go",      "Go's goroutines and channels achieve concurrency with simple syntax. Unlike asyncio coroutines, goroutines are preemptively scheduled by the Go runtime across OS threads. The select statement waits on multiple channel operations, analogous to asyncio's gather for concurrent async operations."),
    ("x_async_rx",      "Reactive Extensions (Rx) model asynchronous data streams as Observables. Operators like map, filter, and flatMap compose asynchronous pipelines declaratively. RxPY brings this paradigm to Python, complementing asyncio's coroutine-based concurrency with a stream-processing abstraction."),
    ("x_async_trio",    "Trio is an alternative async framework for Python that enforces structured concurrency: all concurrent tasks must be grouped in nurseries, and a nursery exits only when all its tasks complete. This eliminates the risk of abandoned tasks that asyncio's gather/create_task can produce."),
    ("x_async_aiohttp", "aiohttp is an asynchronous HTTP client/server framework built on asyncio. It uses ClientSession for connection pooling and persistent TCP connections. Server-side, it routes requests to handlers that are async def functions, integrating naturally with asyncio's event loop."),
    # Go confounders
    ("x_go_rust",       "Rust is a systems programming language focused on memory safety and zero-cost abstractions. Like Go, it compiles to native code without a garbage collector, but Rust uses an ownership model rather than a runtime GC. Rust's async ecosystem uses the async/await syntax with executor runtimes like Tokio."),
    ("x_go_erlang",     "Erlang achieves concurrency through lightweight processes (not OS threads) and message passing via mailboxes. Its actor model resembles Go's goroutines and channels: both provide lightweight concurrency primitives, but Erlang's hot-code reloading and let-it-crash philosophy suit fault-tolerant telecom systems."),
    ("x_go_thread",     "Java's virtual threads, introduced in JDK 21 as Project Loom, allow millions of lightweight threads managed by the JVM rather than OS. Like Go goroutines, virtual threads block cheaply. The JVM scheduler mounts virtual threads on platform threads, enabling synchronous-looking code with asynchronous scalability."),
    ("x_go_csp",        "Communicating Sequential Processes (CSP) is the formal model underlying Go's concurrency design. Tony Hoare introduced CSP in 1978 as a way to describe concurrent systems. Go channels are CSP channels; goroutines are CSP processes. Erlang's actor model and Go's CSP model are both approaches to safe concurrent communication."),
    ("x_go_gc",         "Go's garbage collector uses a tri-colour mark-and-sweep algorithm with concurrent marking to minimise stop-the-world pauses. The write barrier ensures the invariant that grey objects do not directly reference white objects during marking. Typical GC pauses in Go 1.21 are under 1 millisecond for most applications."),
    # Rust confounders
    ("x_rust_cpp",      "C++ manual memory management requires explicit new/delete calls and careful ownership tracking. Unlike Rust's compile-time ownership checker, C++ relies on programmer discipline and tools like AddressSanitizer to detect use-after-free and double-free bugs. Smart pointers (unique_ptr, shared_ptr) automate some deallocation."),
    ("x_rust_zig",      "Zig is a systems programming language that provides manual memory control without hidden allocations. Like Rust, Zig has no garbage collector and emphasises safety. However, Zig uses explicit allocator arguments rather than an ownership type system, making it simpler but requiring runtime checks for safety."),
    ("x_rust_swift",    "Swift's ARC (Automatic Reference Counting) manages memory by tracking strong and weak references at compile time and inserting retain/release calls. Unlike Rust's ownership system, ARC can leak memory through retain cycles, which weak references break. Swift's optionals and Result types handle errors without exceptions."),
    ("x_rust_borrow",   "Clang's ThreadSanitizer detects data races in C and C++ by instrumenting memory accesses and synchronisation operations. Like Rust's borrow checker, it finds concurrent mutation bugs, but at runtime rather than compile time, adding 5-15x overhead. Rust's compile-time guarantee makes TSan unnecessary for safe Rust code."),
    ("x_rust_wasm",     "WebAssembly (WASM) is a binary instruction format for a stack-based virtual machine that runs at near-native speed in browsers. Rust is a popular source language for WASM due to its small runtime and predictable performance. wasm-bindgen and wasm-pack automate the JavaScript/WASM interface generation."),
    # TypeScript confounders
    ("x_ts_flow",       "Flow is a static type checker for JavaScript developed by Meta. Like TypeScript, it adds type annotations to JavaScript and catches type errors at compile time. Flow uses a different type inference algorithm and supports some features TypeScript lacks, such as variance annotations on type parameters."),
    ("x_ts_js",         "JavaScript is a dynamically typed language that runs in browsers and Node.js. TypeScript is a strict superset of JavaScript, so all valid JavaScript is valid TypeScript. Migrating to TypeScript incrementally is possible by adding type annotations file by file while keeping the JavaScript build working."),
    ("x_ts_elm",        "Elm is a purely functional language that compiles to JavaScript for building web UIs. Like TypeScript, it adds static types to the frontend, but with stronger guarantees: Elm programs cannot throw runtime exceptions if they typecheck. Its Hindley-Milner type inference requires fewer explicit annotations than TypeScript."),
    ("x_ts_dart",       "Dart is a statically typed language developed by Google that compiles to JavaScript for web and to native ARM code for mobile via Flutter. Like TypeScript, Dart adds sound static typing to a dynamically typed language tradition, supports generics, and has a class-based object model."),
    ("x_ts_pytype",     "mypy is the standard static type checker for Python, reading PEP 484 type hints and verifying consistency. Like TypeScript for JavaScript, mypy adds an optional static type layer to a dynamically typed language without changing runtime behaviour. Both tools support gradual typing: you annotate code incrementally."),
    # Backprop confounders
    ("x_bp_autograd",   "Automatic differentiation (autograd) computes exact gradients by tracking operations in a computation graph. PyTorch's autograd engine records operations in a dynamic computational graph during the forward pass and uses reverse-mode AD during backward. Unlike numerical differentiation, autograd is exact and scales to millions of parameters."),
    ("x_bp_optim",      "Adam optimiser combines momentum (first moment) and adaptive learning rates (second moment) to accelerate gradient descent. It maintains running averages of gradients and squared gradients, correcting for initialisation bias. Adam often converges faster than vanilla SGD, especially for transformer and recurrent network training."),
    ("x_bp_vanish",     "The vanishing gradient problem occurs when gradients become exponentially small as they propagate through many layers during backpropagation, making early layers learn very slowly. Batch normalisation, residual connections, and careful weight initialisation (He, Xavier) mitigate the problem. ReLU activations reduce vanishing compared to sigmoid."),
    ("x_bp_loss",       "Cross-entropy loss measures the divergence between predicted probability distributions and one-hot target labels. For classification tasks, it equals negative log-likelihood of the correct class. Minimising cross-entropy during training is equivalent to maximum likelihood estimation of the model parameters given the data."),
    ("x_bp_reg",        "Dropout is a regularisation technique that randomly sets neuron activations to zero during training with probability p, preventing co-adaptation of neurons. At inference, all neurons are active and activations are scaled by (1-p). Dropout approximates training an ensemble of exponentially many sparse networks."),
    # Transformer confounders
    ("x_tf_rnn",        "Recurrent neural networks (RNNs) process sequences step by step, maintaining a hidden state that carries information across time steps. LSTMs and GRUs use gating mechanisms to control information flow, mitigating the vanishing gradient problem. However, sequential computation prevents parallelisation across time steps."),
    ("x_tf_bert",       "BERT (Bidirectional Encoder Representations from Transformers) pre-trains a transformer encoder on masked language modelling and next sentence prediction. Fine-tuning on labelled datasets achieves state-of-the-art on question answering, sentiment analysis, and named entity recognition. BERT uses WordPiece tokenisation."),
    ("x_tf_attn",       "Attention mechanisms allow neural networks to focus on relevant parts of the input when producing each output token. Bahdanau attention, introduced for machine translation, computes a context vector as a weighted sum of encoder hidden states. Dot-product attention scales similarity by the square root of key dimension."),
    ("x_tf_gpt",        "GPT models are autoregressive transformer decoders trained to predict the next token given a context. Unlike BERT's bidirectional encoder, GPT uses a causal attention mask so each position attends only to preceding tokens. GPT-3's few-shot learning demonstrated that large language models can perform tasks with minimal examples."),
    ("x_tf_vit",        "Vision Transformers (ViT) apply transformer architectures to image recognition by splitting images into fixed-size patches, projecting them to embeddings, and processing with standard transformer layers. ViT matches or surpasses CNNs when trained on large datasets, demonstrating transformers' generality across modalities."),
    # SVM confounders
    ("x_svm_lr",        "Logistic regression is a linear classifier that models the probability of a binary outcome using the logistic sigmoid function. Unlike SVM, it minimises log-loss rather than maximising margin, producing probabilistic outputs. L1 regularisation induces sparsity; L2 regularisation (ridge) shrinks coefficients."),
    ("x_svm_kernel",    "The polynomial kernel k(x, y) = (x·y + c)^d computes inner products in a polynomial feature space without explicitly constructing the features. Used in SVMs, it enables non-linear classification by implicitly mapping data to higher dimensions. Degree d and coefficient c are hyperparameters tuned by cross-validation."),
    ("x_svm_forest",    "Random forests train an ensemble of decision trees on bootstrap samples using a random subset of features at each split. The ensemble prediction is the majority vote (classification) or mean (regression) of all trees. Feature importance is estimated by the average decrease in impurity across all trees."),
    ("x_svm_svc",       "The hinge loss function used in SVM training penalises misclassified points and correctly classified points within the margin. The dual formulation of SVM expresses the problem in terms of support vectors: only training points near the decision boundary affect the final hyperplane, making SVMs memory-efficient at inference."),
    ("x_svm_pca",       "Principal component analysis (PCA) projects data onto the directions of maximum variance by computing the eigenvectors of the covariance matrix. It reduces dimensionality while preserving most information. Unlike SVM, PCA is an unsupervised method used for preprocessing rather than classification."),
    # K-means confounders
    ("x_km_dbscan",     "DBSCAN is a density-based clustering algorithm that groups points within a minimum density threshold into clusters and labels low-density points as noise. Unlike k-means, DBSCAN discovers the number of clusters automatically and handles arbitrarily shaped clusters. It requires two hyperparameters: epsilon radius and min_samples."),
    ("x_km_gmm",        "Gaussian mixture models (GMM) generalise k-means by modelling each cluster as a Gaussian distribution with a full covariance matrix, allowing ellipsoidal clusters. The EM algorithm iterates between soft cluster assignment (E-step) and parameter update (M-step). GMM produces probabilistic cluster assignments unlike k-means' hard assignments."),
    ("x_km_init",       "K-means++ selects initial cluster centroids with probability proportional to the squared distance from the nearest existing centroid. This contrasts with random initialisation, which can produce poor clusterings. K-means++ guarantees an O(log k) approximation ratio and typically requires fewer iterations to converge."),
    ("x_km_spectral",   "Spectral clustering uses eigenvectors of the graph Laplacian matrix to embed data into a low-dimensional space before applying k-means. It can discover non-convex clusters that k-means fails on. The graph is typically a k-nearest-neighbour graph weighted by Gaussian kernel similarity."),
    ("x_km_elbow",      "The elbow method selects the number of k-means clusters by plotting within-cluster sum of squares (inertia) against k and finding the inflection point where additional clusters yield diminishing returns. The silhouette score is an alternative metric measuring how similar points are to their own cluster versus neighbouring clusters."),
    # XGBoost confounders
    ("x_xgb_lgbm",      "LightGBM is a gradient boosting framework by Microsoft that uses histogram-based split finding and leaf-wise tree growth rather than XGBoost's level-wise growth. It trains faster and uses less memory on large datasets. GOSS (Gradient-based One-Side Sampling) and EFB (Exclusive Feature Bundling) further reduce computation."),
    ("x_xgb_catboost",  "CatBoost handles categorical features natively using ordered boosting and target statistics encoding. Unlike XGBoost which requires one-hot encoding of categoricals, CatBoost learns embeddings during training. It uses symmetric decision trees for fast and reproducible predictions."),
    ("x_xgb_early",     "Early stopping in gradient boosting monitors a validation metric after each boosting round and stops training when the metric stops improving for a specified number of rounds. This prevents overfitting without tuning the number of trees explicitly. XGBoost, LightGBM, and CatBoost all support early stopping natively."),
    ("x_xgb_shap",      "SHAP (SHapley Additive exPlanations) values explain individual predictions by attributing each feature's contribution using cooperative game theory. TreeExplainer computes exact SHAP values for tree models like XGBoost in polynomial time. SHAP values satisfy desirable axioms: local accuracy, missingness, and consistency."),
    ("x_xgb_hyper",     "Hyperparameter tuning for gradient boosting models involves searching over learning rate, max depth, subsample ratio, column subsample ratio, and regularisation terms. Bayesian optimisation with tools like Optuna models the objective as a Gaussian process and selects candidate hyperparameters with high expected improvement."),
    # PostgreSQL confounders
    ("x_pg_mysql",      "MySQL uses a row-level locking model and the InnoDB storage engine for transactional tables. Unlike PostgreSQL's MVCC which avoids locking for readers, MySQL's MVCC implementation still uses undo logs in the shared tablespace. MySQL replication uses binary logs that capture row or statement changes for replica streaming."),
    ("x_pg_index",      "PostgreSQL supports multiple index types: B-tree (default, for ordered data), Hash (equality only), GiST (geometric, full-text), GIN (inverted index for arrays and JSONB), BRIN (block-range for sequential data), and SP-GiST (space-partitioned). Partial and expression indexes target specific rows or computed values."),
    ("x_pg_vacuum",     "VACUUM in PostgreSQL reclaims storage occupied by dead tuples left by MVCC. VACUUM FULL rewrites the entire table to recover all dead space but requires an exclusive lock. Autovacuum automatically runs VACUUM and ANALYZE based on tuple death rates. The visibility map tracks pages with all-visible tuples to speed up vacuum."),
    ("x_pg_json",       "PostgreSQL's JSONB type stores JSON in a decomposed binary format that supports indexing via GIN indexes on keys and values. The @> operator tests JSON containment; ->> extracts text values; #> navigates nested paths. JSONB supports partial updates with jsonb_set() and can be validated against JSON Schema."),
    ("x_pg_repl",       "PostgreSQL logical replication streams decoded row changes (INSERT, UPDATE, DELETE) to replica databases using a publication/subscription model. Unlike physical replication which copies WAL bytes, logical replication supports selective table replication and cross-version replication. pglogical extends this for bidirectional and conflict resolution."),
    # MongoDB confounders
    ("x_mongo_index",   "MongoDB indexes are B-tree structures on one or more fields. Compound indexes support equality, range, and sort operations together. Partial indexes index only documents matching a filter expression, reducing size. TTL indexes automatically delete documents after an expiration time, useful for session and log data."),
    ("x_mongo_shard",   "MongoDB sharding distributes collections across shards using a shard key. Hash sharding distributes documents evenly; range sharding groups adjacent values on the same shard for range queries. The query router (mongos) routes operations to the appropriate shard; the config server stores cluster metadata."),
    ("x_mongo_txn",     "MongoDB multi-document transactions provide ACID guarantees across multiple documents and collections within a replica set. Transactions use snapshot isolation: reads see a consistent snapshot at transaction start. Distributed transactions extend this to sharded clusters using two-phase commit coordinated by the config server."),
    ("x_mongo_coll",    "Couchbase is a distributed NoSQL document database that combines MongoDB-style JSON documents with a key-value store and full SQL++ query language. Like MongoDB, it uses a flexible schema and horizontal scaling. Its memory-first architecture caches active documents in RAM for microsecond-latency key-value operations."),
    ("x_mongo_atlas",   "DynamoDB is a fully managed NoSQL service on AWS with single-digit millisecond latency. Like MongoDB, it stores JSON-like documents but with a strict requirement for a partition key and optional sort key. DynamoDB uses consistent hashing for data distribution and provides on-demand or provisioned throughput capacity."),
    # Redis confounders
    ("x_redis_memcache","Memcached is a distributed in-memory caching system that stores arbitrary data as key-value strings with a size limit of 1 MB per value. Unlike Redis, Memcached does not support persistence, pub/sub messaging, or complex data structures beyond strings. It uses a simple LRU eviction policy across a flat key space."),
    ("x_redis_sentinel","Redis Sentinel provides high availability by monitoring Redis instances, promoting replicas to primary on failure, and notifying clients of topology changes. Sentinels communicate via pub/sub and Raft-like voting for leader election. At least three sentinels are recommended to form a quorum and avoid split-brain."),
    ("x_redis_cluster", "Redis Cluster partitions keyspace across 16384 hash slots, distributed among nodes. Keys are assigned to slots by CRC16(key) mod 16384. Each node holds a subset of slots and their replicas. Clients can connect to any node; a MOVED redirect tells the client the correct node for a key hash slot."),
    ("x_redis_persist", "Redis persistence options: RDB snapshots write a point-in-time copy to disk periodically; AOF (Append-Only File) logs every write command. AOF with fsync=everysec provides durability with at most one second of data loss. RDB+AOF hybrid mode writes AOF on top of recent RDB snapshots for fast restart."),
    ("x_redis_types",   "Valkey is a Redis-compatible open-source fork created after Redis 7.4 changed its licence. Like Redis, it supports strings, lists, sets, sorted sets, hashes, streams, and pub/sub. Valkey is maintained by the Linux Foundation and aims for binary compatibility with Redis clients."),
    # Cassandra confounders
    ("x_cass_dyn",      "Amazon DynamoDB offers eventual or strongly consistent reads, tunable at the request level. Like Cassandra, it uses consistent hashing and replication across availability zones. However, DynamoDB is fully managed with no operational overhead; Cassandra requires managing nodes, compaction, and repair."),
    ("x_cass_scylla",   "ScyllaDB is a C++ reimplementation of the Cassandra CQL protocol with a shared-nothing, shard-per-core architecture. It achieves lower latency than Cassandra by eliminating JVM garbage collection and using the Seastar asynchronous framework. ScyllaDB is binary-compatible with Cassandra drivers and tools."),
    ("x_cass_ring",     "Riak is a distributed key-value store inspired by Amazon Dynamo, using consistent hashing to distribute data across a ring of virtual nodes. Like Cassandra, it offers tunable consistency, masterless replication, and eventual consistency by default. Riak uses vector clocks to detect and resolve concurrent writes."),
    ("x_cass_cql",      "HBase is a column-family NoSQL database built on top of Hadoop HDFS, inspired by Google Bigtable. Unlike Cassandra, HBase has a master-slave architecture where RegionServers handle reads/writes and the Master manages metadata. HBase is optimised for random reads/writes on large datasets stored in HDFS."),
    ("x_cass_repair",   "Cockroach DB is a distributed SQL database providing strong consistency using the Raft consensus protocol for replication. Unlike Cassandra's eventual consistency model, CockroachDB provides serialisable isolation across all nodes. It automatically rebalances data across nodes using a range-based sharding scheme."),
    # Elasticsearch confounders
    ("x_es_solr",       "Apache Solr is a search platform built on Lucene, similar to Elasticsearch. Both use inverted indexes for full-text search and support faceting, highlighting, and spell correction. Solr uses ZooKeeper for cluster coordination in SolrCloud; Elasticsearch has a built-in cluster management layer."),
    ("x_es_opensearch", "OpenSearch is an open-source fork of Elasticsearch 7.10, created by AWS after Elastic changed its licence. It maintains API compatibility with Elasticsearch and adds features like Security Analytics, ML-based anomaly detection, and Vector Database capabilities for semantic search using k-NN indexes."),
    ("x_es_inverted",   "An inverted index maps each unique term to the list of documents containing it, enabling fast full-text search. The index stores term frequencies and positions for relevance scoring (BM25) and phrase queries. Lucene, which underlies both Elasticsearch and Solr, uses segment-based immutable inverted indexes that are merged periodically."),
    ("x_es_vector",     "pgvector is a PostgreSQL extension that adds a vector data type and similarity search operators. It supports approximate nearest neighbour search using IVFFlat and HNSW indexes for high-dimensional embeddings. Unlike Elasticsearch's k-NN search, pgvector runs within the PostgreSQL query planner, enabling hybrid SQL and vector queries."),
    ("x_es_kibana",     "Kibana is the visualisation front-end for the Elastic Stack. It provides dashboards, time-series charts, geo maps, and the Discover interface for ad-hoc log exploration. Kibana's Lens tool allows drag-and-drop chart building on Elasticsearch aggregation results without writing queries."),
    # AES confounders
    ("x_aes_3des",      "Triple DES (3DES) applies the DES cipher three times to each 64-bit block using two or three independent 112-bit or 168-bit keys. It was designed as a stopgap after DES's 56-bit keys became brute-forceable. AES superseded 3DES due to its longer block size (128-bit), faster hardware implementations, and stronger security margins."),
    ("x_aes_gcm",       "AES-GCM (Galois/Counter Mode) is an authenticated encryption mode that provides confidentiality and integrity simultaneously. It combines AES-CTR encryption with GHASH authentication. The authentication tag (128-bit) detects tampering; the nonce must be unique per encryption under the same key to preserve security."),
    ("x_aes_hist",      "The Advanced Encryption Standard was selected by NIST in 2001 following a 5-year public competition. Fifteen algorithms were submitted; Rijndael, designed by Joan Daemen and Vincent Rijmen, was selected for its security, efficiency, and flexibility. It replaced DES as the federal government encryption standard."),
    ("x_aes_hw",        "Modern x86 processors include AES-NI hardware instructions that perform AES round operations in a single clock cycle. This reduces AES-128 encryption throughput to 1-2 cycles per byte on modern CPUs, eliminating the performance advantage that stream ciphers previously had over block ciphers for high-throughput applications."),
    ("x_aes_mode",      "Cipher block chaining (CBC) mode XORs each plaintext block with the previous ciphertext block before encryption, creating dependencies between blocks. The initialisation vector (IV) must be unpredictable for semantic security. CBC requires padding for non-block-aligned messages and is vulnerable to padding oracle attacks if error messages leak."),
    # RSA confounders
    ("x_rsa_ecdsa",     "ECDSA (Elliptic Curve Digital Signature Algorithm) provides equivalent security to RSA with much shorter keys: ECDSA-256 matches RSA-3072 in security level. It uses a elliptic curve group rather than integer factorisation. Shorter keys mean faster key generation, signing, and smaller certificate sizes."),
    ("x_rsa_dh",        "Diffie-Hellman key exchange allows two parties to establish a shared secret over a public channel without transmitting the secret itself. Each party generates a private scalar and a corresponding public group element; they exchange public elements and compute the same shared secret. ECDH uses elliptic curve groups for efficiency."),
    ("x_rsa_padding",   "RSA-OAEP (Optimal Asymmetric Encryption Padding) is the recommended RSA encryption padding scheme. It uses a random seed and a mask generation function to produce deterministic but unguessable ciphertext. PKCS#1 v1.5 padding is deprecated due to Bleichenbacher's padding oracle attack against SSL/TLS implementations."),
    ("x_rsa_sign",      "Digital signatures use asymmetric cryptography to prove message authenticity and integrity. The signer hashes the message and encrypts the hash with their private key. Verifiers decrypt with the public key and compare hashes. RSA-PSS and ECDSA are the recommended signature schemes for modern applications."),
    ("x_rsa_cert",      "X.509 certificates bind a public key to an identity, signed by a Certificate Authority (CA). The certificate chain links the end-entity certificate to a trusted root CA via intermediate CAs. Certificate Transparency logs provide an auditable record of all issued certificates to detect mis-issuance."),
    # TLS confounders
    ("x_tls_dtls",      "DTLS (Datagram TLS) adapts TLS to run over UDP, adding sequence numbers and retransmission to handle packet loss. It is used for VoIP, WebRTC media, and game protocols that cannot tolerate TCP's head-of-line blocking. DTLS handshake includes a cookie exchange to prevent DoS amplification attacks."),
    ("x_tls_mtls",      "Mutual TLS (mTLS) extends TLS by requiring the client to present a certificate in addition to the server. Both parties authenticate each other using public key infrastructure. mTLS is used in service meshes like Istio to establish zero-trust authentication between microservices without application-level credentials."),
    ("x_tls_cipher",    "TLS 1.3 reduced the cipher suite list to five: AES-128-GCM-SHA256, AES-256-GCM-SHA384, CHACHA20-POLY1305-SHA256, AES-128-CCM-SHA256, and AES-128-CCM-8-SHA256. All use AEAD (Authenticated Encryption with Associated Data), eliminating the MAC-then-encrypt construction vulnerable to padding oracle attacks."),
    ("x_tls_hsts",      "HTTP Strict Transport Security (HSTS) instructs browsers to connect to a domain only over HTTPS for a specified duration. The max-age directive sets the expiry in seconds; includeSubDomains applies the policy to all subdomains. HSTS preloading hardcodes domains into browsers to prevent initial HTTP redirects."),
    ("x_tls_sni",       "Server Name Indication (SNI) is a TLS extension that allows a client to specify the hostname it is connecting to during the handshake, before the certificate is sent. This enables virtual hosting of multiple TLS sites on a single IP address. Encrypted Client Hello (ECH) encrypts the SNI to prevent network surveillance."),
    # XSS confounders
    ("x_xss_csrf",      "CSRF (Cross-Site Request Forgery) tricks authenticated users into submitting unintended requests to a web application. Unlike XSS which injects scripts, CSRF abuses the browser's automatic cookie inclusion on cross-origin requests. SameSite cookie attribute (Strict or Lax) and CSRF tokens mitigate this attack."),
    ("x_xss_dom",       "DOM-based XSS occurs when client-side JavaScript writes attacker-controlled data to a dangerous sink without sanitisation. Sources include location.hash, document.referrer, and URL parameters; dangerous sinks include innerHTML, document.write(), and eval(). DOM sanitisation libraries like DOMPurify prevent injection."),
    ("x_xss_csp",       "Content Security Policy (CSP) is an HTTP response header that restricts which scripts, styles, and resources a page can load. The default-src directive sets a fallback; script-src restricts JavaScript origins. 'nonce-based' or 'hash-based' CSP allows specific inline scripts while blocking injected ones."),
    ("x_xss_encode",    "HTML entity encoding converts special characters to their HTML equivalents: < to &lt;, > to &gt;, & to &amp;, \" to &quot;. This prevents browser interpretation of injected HTML. Context-sensitive encoding is required: HTML encoding for HTML context, URL encoding for URL parameters, JavaScript encoding for JS strings."),
    ("x_xss_samy",      "The Samy worm (2005) was a self-propagating XSS worm that spread across MySpace profiles by exploiting stored XSS in profile pages. Within 20 hours it had infected over one million profiles. The attack demonstrated the large-scale impact of stored XSS and accelerated adoption of output encoding standards."),
    # SQL injection confounders
    ("x_sqli_nosql",    "NoSQL injection attacks exploit improper handling of operator expressions in NoSQL query languages. MongoDB queries can be manipulated by injecting JSON operators like $gt or $regex if user input is passed directly to query objects. Proper input validation and schema enforcement prevent NoSQL injection."),
    ("x_sqli_blind",    "Blind SQL injection occurs when an application does not return query results in the response but shows different behaviour (true/false responses or timing differences) based on query evaluation. Boolean-based blind injection infers data one bit at a time; time-based injection uses SLEEP() or BENCHMARK() to exfiltrate data."),
    ("x_sqli_stored",   "Second-order SQL injection stores malicious payloads that are initially safely handled but later used in a vulnerable SQL query without proper encoding. For example, a username containing SQL syntax may be safely inserted but unsafely used later in a dynamic query that updates user records."),
    ("x_sqli_union",    "UNION-based SQL injection appends a UNION SELECT to the original query to retrieve data from other tables. The injected SELECT must have the same number of columns and compatible data types. Attackers use ORDER BY or UNION NULL, NULL, NULL... to determine the number of columns in the original query."),
    ("x_sqli_orm",      "Object-Relational Mappers (ORMs) like SQLAlchemy, Hibernate, and Django ORM generate parameterised queries by default, preventing SQL injection. However, raw query interfaces (execute() with string interpolation) bypass this protection. ORM usage alone does not guarantee safety if developers use raw SQL for complex queries."),
    # Kubernetes confounders
    ("x_k8s_argo",      "Argo Workflows is a Kubernetes-native workflow engine for orchestrating parallel jobs. Unlike Kubernetes' built-in Job and CronJob resources, Argo Workflows supports DAG-structured pipelines, conditional execution, and retry logic. Argo CD uses GitOps principles to continuously reconcile cluster state with Git repositories."),
    ("x_k8s_hpa",       "The Kubernetes Horizontal Pod Autoscaler scales Deployment replicas based on observed CPU utilisation or custom metrics from Prometheus via the Metrics API. The HPA controller reconciles the desired replica count every 15 seconds. Vertical Pod Autoscaler (VPA) adjusts resource requests for individual pods instead."),
    ("x_k8s_etcd",      "etcd is the distributed key-value store used by Kubernetes to store all cluster state: nodes, pods, services, secrets, and ConfigMaps. It uses the Raft consensus protocol to provide strong consistency across an odd number of etcd members. etcd is the single source of truth for the Kubernetes control plane."),
    ("x_k8s_cni",       "Kubernetes Container Network Interface (CNI) plugins implement pod networking. Calico uses BGP for route distribution; Cilium uses eBPF for policy enforcement and load balancing; Flannel creates an overlay network with VXLAN. Each pod gets a unique IP; the CNI plugin ensures cross-node reachability."),
    ("x_k8s_rbac",      "Kubernetes RBAC controls access to the API server using Roles (namespace-scoped) and ClusterRoles (cluster-scoped) bound to users, groups, or ServiceAccounts. The principle of least privilege is enforced by granting only the verbs (get, list, create, delete) on the specific resources a workload needs."),
    # Docker confounders
    ("x_docker_oci",    "OCI (Open Container Initiative) standardises container image format and runtime interface. The image specification defines a JSON manifest listing layer digests; layers are tarballs stored in a content-addressable registry. containerd and cri-o implement the OCI runtime spec, allowing Kubernetes to use runtimes beyond Docker."),
    ("x_docker_build",  "Multi-stage Docker builds use multiple FROM instructions to produce a minimal final image. Build stages compile source code or install tools that are not needed at runtime; the final stage copies only the compiled artifacts. This reduces image size by excluding build tools, source code, and intermediate files."),
    ("x_docker_seccomp","Docker applies a default seccomp profile that restricts 44 system calls to reduce the kernel attack surface. The no-new-privileges flag prevents processes from gaining additional privileges via setuid binaries. AppArmor and SELinux profiles provide mandatory access control for container processes."),
    ("x_docker_nspace", "Linux namespaces isolate resources between containers: PID namespace gives each container its own process tree; network namespace provides a private network stack; mount namespace separates filesystem views; user namespace maps container UIDs to unprivileged host UIDs for rootless containers."),
    ("x_docker_compose","Docker Compose defines multi-container applications using a YAML file specifying services, networks, and volumes. compose up starts all services and creates a default bridge network for DNS-based service discovery. Compose is primarily for local development; production deployments use Kubernetes or ECS."),
    # Terraform confounders
    ("x_tf_pulumi",     "Pulumi is an infrastructure-as-code tool that uses general-purpose languages (Python, TypeScript, Go) instead of HCL. Like Terraform, Pulumi manages cloud resources and tracks state. Its programming-language approach enables loops, conditionals, and abstractions that HCL's declarative model cannot express naturally."),
    ("x_tf_ansible",    "Ansible is a configuration management tool that automates server setup by running idempotent playbooks over SSH. Unlike Terraform, which provisions infrastructure declaratively, Ansible configures existing servers imperatively. They are complementary: Terraform provisions; Ansible configures."),
    ("x_tf_module",     "Terraform modules encapsulate reusable infrastructure patterns. A module accepts input variables, creates resources, and exposes output values. The Terraform Registry hosts community and provider-maintained modules for common patterns like VPC networking, EKS clusters, and RDS databases."),
    ("x_tf_drift",      "Infrastructure drift occurs when actual cloud resource configuration diverges from Terraform state, typically due to manual console changes. Terraform plan detects drift by comparing the live resource state (fetched via provider APIs) with the state file. Atlantis and Terraform Cloud enforce plan-before-apply workflows."),
    ("x_tf_provider",   "Terraform providers are plugins that implement resource types for a specific infrastructure platform. The AWS provider communicates with AWS APIs to create EC2 instances, S3 buckets, and VPCs. Providers are versioned in the required_providers block; the provider lock file pins exact versions for reproducibility."),
    # Prometheus confounders
    ("x_prom_grafana",  "Grafana visualises time-series data from Prometheus and other data sources. Dashboards contain panels with PromQL queries; alerts can be configured on panel data. Grafana Loki provides log aggregation with a LogQL query language similar to PromQL for logs, enabling unified observability dashboards."),
    ("x_prom_otel",     "OpenTelemetry (OTel) provides vendor-neutral APIs and SDKs for traces, metrics, and logs. OTel metrics data can be exported to Prometheus via the Prometheus exporter. Unlike Prometheus's pull-based scraping, OTel collectors push telemetry to backends, supporting both Prometheus and OTLP-compatible systems."),
    ("x_prom_counter",  "Prometheus metrics types: Counter (monotonically increasing), Gauge (can increase or decrease), Histogram (samples observations into buckets), Summary (computes quantiles client-side). rate() computes per-second rate of increase of a counter over a time window, correcting for resets."),
    ("x_prom_alert",    "Prometheus Alertmanager routes alerts based on labels, grouping related alerts and silencing or inhibiting them based on routing rules. Receivers (PagerDuty, Slack, OpsGenie) are configured per route. Dead man's switch alerts fire when no heartbeat metric is received, detecting total monitoring failure."),
    ("x_prom_tsdb",     "Prometheus's time-series database stores metrics in 2-hour chunks of compressed blocks. The compaction process merges small blocks into larger ones, applying Gorilla compression (XOR delta encoding for timestamps and values). Remote write allows forwarding samples to long-term storage systems like Thanos or Cortex."),
    # nginx confounders
    ("x_nginx_haproxy", "HAProxy is a high-availability load balancer and proxy server that distributes TCP and HTTP traffic across backend servers. Like nginx, it supports health checks, SSL termination, and connection keep-alive. HAProxy's ACL-based routing engine and detailed statistics interface make it popular for complex load balancing scenarios."),
    ("x_nginx_traefik", "Traefik is a modern reverse proxy and ingress controller designed for cloud-native environments. Unlike nginx, it auto-discovers services from Docker labels, Kubernetes Ingress resources, and Consul catalogs. Traefik automatically provisions TLS certificates via Let's Encrypt using ACME challenges."),
    ("x_nginx_conf",    "nginx configuration uses a block-based syntax with contexts: main, events, http, server, and location. The location block matches URI prefixes or regex patterns and specifies how to handle matching requests. The proxy_pass directive forwards requests to upstream servers; proxy_cache enables response caching."),
    ("x_nginx_limit",   "Rate limiting in nginx uses the limit_req_zone directive to define a shared memory zone tracking request rates per client IP. The limit_req directive applies the zone to specific locations with a burst parameter for bursty traffic. This prevents abuse and ensures fair resource allocation across clients."),
    ("x_nginx_upstream","nginx upstream blocks define groups of backend servers with load balancing methods: round-robin (default), least connections, IP hash, or random. Health checks with max_fails and fail_timeout remove unresponsive servers from rotation. The keepalive directive maintains persistent connections to upstream servers."),
    # TCP confounders
    ("x_tcp_udp",       "UDP is a connectionless, unreliable transport protocol with minimal overhead. Unlike TCP, UDP provides no retransmission, flow control, or congestion control. Applications like DNS, VoIP, and online games use UDP when low latency is more important than guaranteed delivery, implementing reliability at the application layer if needed."),
    ("x_tcp_quic2",     "QUIC implements its own congestion control, defaulting to Cubic or NewReno, with pluggable algorithms. Unlike TCP where congestion control runs in the kernel, QUIC runs in user space, enabling rapid algorithm deployment. ECN (Explicit Congestion Notification) allows routers to signal congestion without packet loss."),
    ("x_tcp_nagle",     "Nagle's algorithm reduces the number of small TCP packets by coalescing data until a full segment can be sent or an acknowledgement arrives. This increases efficiency for chatty protocols but adds latency. TCP_NODELAY disables Nagle's algorithm for latency-sensitive applications like interactive terminals and real-time games."),
    ("x_tcp_syn",       "SYN cookies defend TCP servers against SYN flood DoS attacks. Instead of allocating state for each SYN packet, the server encodes connection state in the SYN-ACK sequence number. When the client's ACK arrives, the server reconstructs state from the cookie, preventing memory exhaustion without maintaining a half-open connection queue."),
    ("x_tcp_timewait",  "TCP TIME_WAIT state holds a connection for 2×MSL (Maximum Segment Lifetime, typically 60 seconds) after graceful close to prevent delayed packets from a closed connection being misinterpreted by a new connection reusing the same 4-tuple. SO_REUSEADDR allows servers to bind a port in TIME_WAIT, reducing restart delays."),
    # DNS confounders
    ("x_dns_doh",       "DNS over HTTPS (DoH) encrypts DNS queries by sending them as HTTPS requests to a resolver, preventing eavesdropping and manipulation. Unlike traditional DNS on UDP port 53, DoH uses port 443 and blends with HTTPS traffic. DNS over TLS (DoT) provides similar privacy on a dedicated port 853."),
    ("x_dns_anycast",   "Anycast routing assigns the same IP address to multiple geographically distributed servers. DNS resolvers like Cloudflare (1.1.1.1) and Google (8.8.8.8) use anycast so queries are routed to the nearest instance by BGP path selection. Anycast provides latency reduction and DDoS resilience."),
    ("x_dns_zone",      "A DNS zone is an administrative delegation of a portion of the DNS namespace. Zone files contain resource records: A (IPv4 address), AAAA (IPv6), CNAME (alias), MX (mail exchange), TXT (text), and SOA (start of authority). Zone transfers (AXFR/IXFR) replicate zones from primary to secondary nameservers."),
    ("x_dns_ttl",       "DNS TTL (Time to Live) specifies how long resolvers cache a record before re-querying the authoritative nameserver. Short TTLs (60-300 seconds) enable rapid IP address changes during deployments or failover; long TTLs (3600-86400 seconds) reduce DNS query load. TTL should be shortened before planned IP changes."),
    ("x_dns_enum",      "DNS enumeration is a reconnaissance technique that queries DNS records to discover subdomains, mail servers, and name servers of a target domain. Zone transfer vulnerabilities allow transferring all DNS records. DMARC (Domain-based Message Authentication) records prevent email spoofing by specifying authorised sending infrastructure."),
    # HTTP/2 confounders
    ("x_http2_http3",   "HTTP/3 replaces TCP with QUIC, eliminating TCP's head-of-line blocking even at the transport layer. Unlike HTTP/2's multiplexing over a single TCP stream, HTTP/3 streams are fully independent: packet loss on one stream does not delay others. The QUIC handshake combines transport and TLS negotiation, reducing connection setup latency."),
    ("x_http2_rest",    "REST (Representational State Transfer) is an architectural style for web services using HTTP methods (GET, POST, PUT, DELETE) to perform CRUD operations on resources identified by URLs. RESTful APIs return JSON or XML responses. HTTP/2 improves REST performance by multiplexing multiple API requests over a single connection."),
    ("x_http2_grpc",    "gRPC uses HTTP/2 for transport, Protocol Buffers for serialisation, and supports four communication patterns: unary, server streaming, client streaming, and bidirectional streaming. HTTP/2 multiplexing is critical for gRPC's performance: a single connection handles thousands of concurrent RPC streams."),
    ("x_http2_push",    "HTTP/2 server push was designed to reduce latency by proactively sending resources the server predicts the client will request. However, push proved hard to implement correctly (servers lack client cache state) and was removed from HTTP/3. Resource hints (preload, prefetch) at the HTML level are more practical alternatives."),
    ("x_http2_hol",     "Head-of-line blocking occurs when the first packet in a sequence must be processed before subsequent packets can proceed. HTTP/1.1 pipelining suffers HOL blocking at the application layer. HTTP/2 eliminates application-layer HOL blocking via multiplexing but retains TCP transport-layer HOL blocking, which HTTP/3 QUIC resolves."),
    # BGP confounders
    ("x_bgp_ospf",      "OSPF (Open Shortest Path First) is a link-state routing protocol used within autonomous systems. Unlike BGP's path-vector algorithm, OSPF uses Dijkstra's SPF algorithm on a complete topology map. OSPF routers flood Link State Advertisements to maintain a consistent view of the network and compute loop-free shortest paths."),
    ("x_bgp_hijack",    "BGP hijacking occurs when a router announces a more specific or equal IP prefix than the legitimate owner, attracting traffic to the wrong AS. The 2008 Pakistan Telecom incident accidentally blackholed YouTube globally for two hours. RPKI (Resource Public Key Infrastructure) cryptographically validates prefix origin to prevent hijacking."),
    ("x_bgp_comm",      "BGP communities are 32-bit tags attached to routes to encode policy information. Well-known communities include NO_EXPORT (don't advertise to eBGP peers) and NO_ADVERTISE (don't advertise to any peer). Large communities (RFC 8092) extend to 96 bits, enabling more expressive routing policy signalling between ASes."),
    ("x_bgp_evpn",      "EVPN (Ethernet VPN) uses BGP as the control plane for Layer 2 and Layer 3 VPNs in data centre fabrics. BGP EVPN route types carry MAC/IP bindings, IP prefixes, and multicast group membership. VXLAN is commonly used as the data plane encapsulation, with BGP distributing VTEP endpoints."),
    ("x_bgp_flowspec",  "BGP Flowspec (RFC 5575) extends BGP to distribute traffic filtering rules, enabling network-wide DoS mitigation by disseminating rate-limit or drop rules from a controller to edge routers. Flowspec rules match on destination prefix, source prefix, protocol, port ranges, and DSCP values."),
    # QUIC confounders
    ("x_quic_webtrans", "WebTransport is a web API built on HTTP/3 that enables bidirectional, multiplexed communication between browsers and servers. Unlike WebSockets (which run over TCP), WebTransport uses QUIC streams and datagrams, enabling unreliable datagrams alongside reliable streams in a single connection."),
    ("x_quic_wireguard","WireGuard is a modern VPN protocol that runs in the Linux kernel and uses state-of-the-art cryptography (ChaCha20 for encryption, Curve25519 for key exchange). Like QUIC, it runs over UDP and has a simpler implementation than legacy VPN protocols. WireGuard does not use TLS; it has its own handshake."),
    ("x_quic_mptcp",    "Multipath TCP (MPTCP) extends TCP to use multiple network paths simultaneously. A single data stream is split across subflows (e.g., Wi-Fi and cellular), increasing throughput and providing seamless handover. Like QUIC's connection migration, MPTCP can survive changes in network interface."),
    ("x_quic_sctp",     "SCTP (Stream Control Transmission Protocol) provides multi-stream, multi-homing transport over IP. It was designed for telephony signalling but shares QUIC's goals: multiple independent streams, connection migration, and resistance to SYN flood. SCTP never achieved widespread adoption due to middlebox incompatibility."),
    ("x_quic_udp",      "UDP is the transport layer protocol underlying QUIC. QUIC runs entirely in user space, implemented in application code rather than the OS kernel, which enabled rapid iteration. QUIC packets are encrypted at the packet level, preventing middleboxes from inspecting or modifying them, unlike TCP whose headers are visible."),
    # Dijkstra confounders
    ("x_dijk_bellman",  "Bellman-Ford computes shortest paths in graphs that may contain negative edge weights, which Dijkstra cannot handle. It relaxes all edges V-1 times, detecting negative cycles on the V-th relaxation. Time complexity is O(VE). The SPFA optimisation uses a queue to process only recently updated vertices, improving average-case performance."),
    ("x_dijk_astar",    "A* search extends Dijkstra's algorithm with a heuristic function h(n) estimating the cost from node n to the goal. It expands nodes in order of f(n) = g(n) + h(n), where g(n) is the cost from start. With an admissible (non-overestimating) heuristic, A* is optimal. It is widely used for pathfinding in games and maps."),
    ("x_dijk_floyd",    "Floyd-Warshall computes all-pairs shortest paths in O(V³) using dynamic programming. It iteratively improves paths by considering each vertex as an intermediate node. It handles negative edges (but not negative cycles) and can detect negative cycles by checking diagonal entries. Useful when V is small and all-pairs distances are needed."),
    ("x_dijk_johnson",  "Johnson's algorithm computes all-pairs shortest paths in sparse graphs more efficiently than Floyd-Warshall. It reweights edges using Bellman-Ford to eliminate negative weights, then runs Dijkstra from each vertex. Time complexity is O(V² log V + VE), better than O(V³) for sparse graphs."),
    ("x_dijk_topo",     "Topological sort orders vertices of a directed acyclic graph (DAG) such that every edge goes from an earlier vertex to a later one. It can be computed in O(V+E) using DFS or Kahn's algorithm (BFS from zero in-degree vertices). Shortest paths in DAGs are found in O(V+E) by relaxing edges in topological order."),
    # Bloom filter confounders
    ("x_bloom_cuckoo",  "Cuckoo filters improve on Bloom filters by supporting deletion and achieving better lookup performance. They store fingerprints of items in a hash table with two possible bucket locations. Insertion displaces existing fingerprints (like the cuckoo bird) if both buckets are full. False-positive rates are comparable to Bloom filters."),
    ("x_bloom_count",   "Counting Bloom filters replace each bit with a counter, enabling element deletion by decrementing counters on removal. They use more space than standard Bloom filters but support a multiset membership test. Overflow of counters (when an element is inserted more than 2^k times) can cause false negatives."),
    ("x_bloom_hll",     "HyperLogLog (HLL) estimates the cardinality of a multiset using O(log log n) space. It hashes each element and tracks the maximum leading zeros in binary representations of hash values. Multiple registers with harmonic mean estimation reduce variance. Redis's PFADD and PFCOUNT commands implement HLL."),
    ("x_bloom_skip",    "Skip lists are probabilistic data structures providing O(log n) average-case search, insert, and delete. They layer linked lists with geometric probability of promotion to higher levels, achieving balanced-tree performance with simpler implementation. Redis sorted sets use skip lists for range queries on scores."),
    ("x_bloom_quotient","Quotient filters offer faster cache performance than Bloom filters by storing quotients (high bits of fingerprints) in a compact hash table. Elements are fingerprinted and stored in a slot array; linear probing resolves collisions. Quotient filters support merging and can be resized without rehashing."),
    # B-tree confounders
    ("x_btree_art",     "The Adaptive Radix Tree (ART) is an in-memory index structure that uses path compression and lazy expansion to achieve O(k) lookup where k is key length. ART adapts node types (4, 16, 48, or 256 children) to actual fanout, achieving space efficiency. It outperforms B-trees for in-memory workloads."),
    ("x_btree_bplus",   "B+ trees extend B-trees by storing all data records only in leaf nodes, with leaf nodes linked in a doubly-linked list. Internal nodes store only keys for routing. This design enables efficient range scans by traversing the leaf list without backtracking. Almost all database index implementations use B+ trees."),
    ("x_btree_fractal", "Fractal cascade indexes (FractalDB, TokuDB) buffer write operations in internal nodes, pushing them down lazily to leaf nodes in batches. This amortises the I/O cost of random writes into sequential batch writes, achieving write throughput orders of magnitude better than B-trees for write-heavy workloads."),
    ("x_btree_rbtree",  "Red-black trees are self-balancing binary search trees that maintain O(log n) height by colouring nodes red or black and enforcing colouring invariants. They are used in Java's TreeMap, C++ std::map, and Linux kernel's completely fair scheduler. Unlike B-trees, they are designed for in-memory use with binary branching."),
    ("x_btree_trie",    "Tries (prefix trees) store strings by decomposing them into characters along tree edges. Each path from root to leaf represents a string; shared prefixes share path segments. Compressed tries (Patricia tries) merge single-child nodes. Tries enable O(k) string operations independent of dataset size."),
    # LSM-tree confounders
    ("x_lsm_wisckey",  "WiscKey separates keys and values in LSM trees: keys remain in the LSM tree for sorted ordering, while values are appended to a value log (vLog). This reduces write amplification by not recopying large values during compaction. Garbage collection reclaims space in the vLog when the corresponding key is deleted."),
    ("x_lsm_rocks",     "RocksDB is an LSM-tree key-value store developed by Facebook, forked from LevelDB. It adds column families for logical data separation, compaction filters, merge operators, and rate limiting. Compaction styles include levelled (space-efficient), universal (write-optimised), and FIFO (time-series data)."),
    ("x_lsm_comp",      "LSM-tree compaction strategies differ in their write amplification, space amplification, and read amplification tradeoffs. Levelled compaction keeps each level's data sorted and non-overlapping, reducing read amplification but increasing write amplification. Tiered/STCS compaction reduces write amplification at the cost of more compaction I/O."),
    ("x_lsm_wal",       "The write-ahead log (WAL) in LSM trees ensures durability by persisting mutations to a sequential log before acknowledging writes. On crash recovery, the WAL is replayed to reconstruct the in-memory memtable. The WAL is truncated after the memtable is flushed to an immutable SSTable."),
    ("x_lsm_bloom2",    "LSM trees use Bloom filters per SSTable to avoid unnecessary disk reads during point lookups. When querying a key, the system checks each SSTable's Bloom filter before reading the file. Most files will have a negative result, preventing expensive random I/O. The false-positive rate is tunable to balance memory and I/O."),
    # Consistent hashing confounders
    ("x_ch_rendezvous", "Rendezvous hashing (highest random weight, HRW) assigns each key to the node with the highest hash(key, node) score. Unlike consistent hashing, there is no ring structure. Adding or removing a node requires comparing all nodes, but it achieves perfect load balance and minimal key movement."),
    ("x_ch_maglev",     "Maglev hashing is a consistent hashing scheme used by Google's load balancer. It builds a permutation table where each backend fills slots in its preferred order, achieving near-perfect load balance and fast lookups via table lookup. Adding backends causes minimal disruption to existing connections."),
    ("x_ch_partition",  "Hash partitioning distributes data by computing hash(key) mod N for N partitions. Unlike consistent hashing, resizing requires rehashing all keys. Some distributed databases use fixed partition counts (e.g., Elasticsearch's 1024 primary shards default) and rebalance by migrating complete shards."),
    ("x_ch_vnodes",     "Virtual nodes (vnodes) solve uneven distribution in consistent hashing by assigning each physical node multiple positions on the ring. With V vnodes per node, the probability that any node owns more than a fair share decreases as V grows. Cassandra uses 256 vnodes per node by default."),
    ("x_ch_jump",       "Jump consistent hash maps keys to buckets in O(1) time and O(1) space using a pseudo-random jump sequence. When the number of buckets increases from n to n+1, exactly 1/(n+1) of keys move, achieving minimal disruption. Unlike ring-based consistent hashing, jump hash has no notion of vnodes."),
    # React Fiber confounders
    ("x_react_vue",     "Vue.js is a progressive JavaScript framework for building UIs. Unlike React's Fiber reconciler, Vue 3 uses a compiled virtual DOM with static tree hoisting to skip diffing of static nodes. Vue's reactivity system tracks dependencies through Proxy objects rather than React's immutable state and explicit re-renders."),
    ("x_react_svelte",  "Svelte compiles components to vanilla JavaScript at build time, eliminating the virtual DOM entirely. Unlike React's runtime reconciler, Svelte's compiled output directly manipulates the DOM when state changes. This produces smaller bundles and eliminates reconciliation overhead at the cost of compile-time processing."),
    ("x_react_signal",  "Solid.js uses fine-grained reactivity through signals: primitive values that notify dependent computations when they change. Unlike React's coarse component re-renders, Solid updates only the DOM nodes that depend on changed signals. Components run once; signal subscriptions handle updates without re-rendering component trees."),
    ("x_react_memo",    "React.memo wraps a component to skip re-rendering when its props are shallowly equal to the previous render. useMemo caches expensive computed values; useCallback memoises function references. These optimisations prevent unnecessary reconciliation for pure components with stable props."),
    ("x_react_state",   "React hooks manage component state and side effects. useState returns a state value and setter; useReducer handles complex state transitions with a reducer function. useEffect runs side effects after renders with a dependency array to control when it re-runs. Custom hooks encapsulate reusable stateful logic."),
    # GraphQL confounders
    ("x_gql_rest",      "REST APIs expose resources at fixed URLs with standard HTTP methods. Unlike GraphQL's single endpoint, REST uses multiple endpoints (one per resource). REST is stateless and leverages HTTP caching; GraphQL queries cannot use GET-based caching as easily. REST is simpler for simple CRUD APIs; GraphQL excels for complex data graphs."),
    ("x_gql_trpc",      "tRPC enables end-to-end typesafe APIs without a schema language. Like GraphQL, it provides type safety between client and server, but uses TypeScript types directly instead of a schema definition language. tRPC is simpler to set up for TypeScript monorepos but requires TypeScript on both client and server."),
    ("x_gql_datalodr",  "DataLoader batches and caches database queries to solve the N+1 query problem in GraphQL resolvers. When resolvers request related entities one at a time, DataLoader collects all requests within a tick and issues a single batched query. This reduces database load from O(N) queries to O(1) per batch."),
    ("x_gql_sub",       "GraphQL subscriptions push real-time updates to clients using WebSockets or server-sent events. The server sends events whenever the subscribed data changes. Apollo Server uses graphql-ws for WebSocket subscriptions. Unlike polling, subscriptions deliver data immediately when it changes with low overhead."),
    ("x_gql_fed",       "GraphQL Federation composes a distributed graph from multiple subgraph services, each owning a portion of the schema. Apollo Gateway stitches subgraphs by stitching @key entities across services. This allows teams to own their subgraphs independently while exposing a unified API to clients."),
    # CORS confounders
    ("x_cors_csp2",     "Content Security Policy (CSP) prevents resource loading from unauthorised origins using HTTP response headers. Unlike CORS which governs cross-origin requests, CSP restricts what resources a page can load. CSP is enforced by the browser even for same-origin content to mitigate XSS attacks."),
    ("x_cors_samesite", "SameSite cookie attribute restricts whether cookies are sent with cross-site requests. Strict mode prevents sending cookies on any cross-site navigation; Lax allows cookies on top-level navigations. None allows all cross-site cookie sending (requires Secure flag). SameSite=Lax is the browser default since Chrome 80."),
    ("x_cors_preflight","CORS preflight uses an HTTP OPTIONS request sent by the browser before a cross-origin request with non-simple methods or headers. The server responds with Access-Control-Allow-Methods and Access-Control-Allow-Headers. If the preflight succeeds, the browser sends the actual request."),
    ("x_cors_proxy",    "A CORS proxy sits between a browser and an API that lacks CORS headers. The proxy adds the required Access-Control-Allow-Origin header to responses, allowing browsers to read the data. CORS proxies are commonly used during development but are a security risk in production as they bypass same-origin policy."),
    ("x_cors_wildcard", "Using wildcard (*) in Access-Control-Allow-Origin permits any origin to make cross-origin requests. This disables protection for resources that should be access-controlled. The wildcard cannot be combined with credentials (cookies, auth headers); credentialed requests require an explicit origin in the Allow-Origin header."),
    # SSR confounders
    ("x_ssr_isr",       "Incremental Static Regeneration (ISR) in Next.js combines static generation with background revalidation. Pages are generated at build time and served as static HTML. When a request arrives after the revalidation interval, Next.js regenerates the page in the background and serves the stale page to the current request."),
    ("x_ssr_astro",     "Astro uses a component island architecture where pages are rendered as static HTML by default. Interactive components (React, Vue, Svelte) are selectively hydrated using the client:load, client:idle, or client:visible directives. This produces minimal JavaScript bundles by hydrating only the interactive parts."),
    ("x_ssr_remix",     "Remix renders HTML on the server using loader functions that fetch data before rendering. Unlike Next.js SSR, Remix handles forms and mutations natively through action functions, enabling progressive enhancement. Nested routes with independent loaders allow parallel data fetching for complex page layouts."),
    ("x_ssr_spa",       "Single-page applications (SPAs) render entirely in the browser using JavaScript. The initial HTML is a shell; content is rendered by client-side JavaScript after loading a bundle. SPAs have poor initial load performance and SEO compared to SSR, but avoid server round trips for subsequent navigation."),
    ("x_ssr_edge",      "Edge rendering executes server-side code at CDN edge locations close to users. Cloudflare Workers and Vercel Edge Functions use V8 isolates rather than Node.js, enabling cold starts under 1 ms. Edge rendering personalises content globally without the latency of centralised origin servers."),
    # WebSocket confounders
    ("x_ws_sse",        "Server-Sent Events (SSE) provide one-directional push from server to browser over a persistent HTTP connection. The EventSource API reconnects automatically. Unlike WebSockets, SSE uses standard HTTP, works through proxies, and is limited to text messages. SSE is simpler for server-to-client streaming like live feeds."),
    ("x_ws_socket_io",  "Socket.IO is a library that abstracts WebSocket communication with automatic fallbacks to long-polling. It adds namespaces, rooms, acknowledgements, and automatic reconnection on top of the WebSocket protocol. Socket.IO uses a custom wire format, so native WebSocket clients cannot connect without the Socket.IO protocol."),
    ("x_ws_longpoll",   "Long polling simulates server push over HTTP by holding a request open until the server has new data to send. The client immediately sends a new request after receiving a response. Long polling has higher latency and overhead than WebSockets due to HTTP request overhead on each message."),
    ("x_ws_signalingr",  "SignalR is Microsoft's real-time communication library for ASP.NET. Like Socket.IO, it abstracts over WebSockets, Server-Sent Events, and long polling, choosing the best available transport. SignalR supports hubs for broadcasting messages to groups of connected clients."),
    ("x_ws_frame",      "WebSocket frames have a 2-14 byte header: a FIN bit, opcode (text, binary, ping, pong, close), masking bit, and payload length (7-bit, 16-bit, or 64-bit). Client-to-server frames must be masked with a random 32-bit key to prevent cache poisoning by intermediaries. Server-to-client frames are unmasked."),
    # mmap confounders
    ("x_mmap_copy",     "Copy-on-write (CoW) is a memory optimisation where forked processes share physical pages mapped to the same virtual addresses. When either process writes to a shared page, the kernel allocates a new physical page and updates the page table. fork() is efficient because it delays copying until modification."),
    ("x_mmap_shmem",    "POSIX shared memory (shm_open, mmap) enables inter-process communication by mapping the same physical memory into multiple process address spaces. Unlike pipes and sockets, shared memory transfers no data between kernel and user space. Synchronisation requires mutexes, semaphores, or futexes."),
    ("x_mmap_hugepage", "Huge pages (2 MB or 1 GB on x86) reduce TLB pressure for workloads with large working sets. Fewer TLB entries cover the same address range, reducing TLB miss penalties. Transparent Huge Pages (THP) automatically promote aligned 2 MB regions to huge pages, but can cause latency spikes during defragmentation."),
    ("x_mmap_fault",    "Page faults occur when a process accesses a virtual address without a current physical mapping. Minor faults allocate a new physical frame; major (hard) faults load data from disk. Memory-mapped I/O uses major faults to transparently load file data on access, avoiding explicit read() calls."),
    ("x_mmap_numa",     "NUMA (Non-Uniform Memory Access) architectures have multiple memory banks with different latencies depending on which CPU socket is accessing them. The Linux kernel's NUMA memory policy (numactl, mbind) controls whether allocations prefer local or interleaved memory. mmap() allocates memory on the NUMA node closest to the allocating CPU."),
    # epoll confounders
    ("x_epoll_kqueue",  "kqueue is the BSD equivalent of Linux's epoll, providing event notification for file descriptors, signals, timers, and processes in a single API. Unlike epoll which uses three system calls (create, ctl, wait), kqueue registers and waits in kevent(). Filters (EVFILT_READ, EVFILT_WRITE) correspond to epoll's EPOLLIN/EPOLLOUT."),
    ("x_epoll_iouring", "io_uring is a Linux asynchronous I/O interface that uses shared ring buffers between kernel and user space to submit and complete I/O operations without system calls for each operation. Unlike epoll, which notifies when I/O is ready, io_uring completes I/O asynchronously, enabling zero-copy networking with IORING_OP_SEND_ZC."),
    ("x_epoll_select",  "select() is the original POSIX multiplexing API with a fixed limit of FD_SETSIZE (typically 1024) file descriptors and O(n) scanning of the descriptor set on each call. poll() removes the limit but still scans all descriptors. Both are obsoleted by epoll for high-connection-count servers."),
    ("x_epoll_et",      "Edge-triggered (EPOLLET) epoll delivers events only when a state change occurs (new data arrives), while level-triggered mode (default) delivers events as long as the condition is true (data is available). Edge-triggered mode requires non-blocking I/O and processing until EAGAIN to avoid missing events."),
    ("x_epoll_thread",  "Reactor pattern uses a single-threaded event loop (epoll/kqueue) to dispatch I/O readiness events to handlers. Proactor pattern (io_uring) initiates asynchronous I/O and dispatches completion events. Thread-per-connection models (Apache prefork) avoid event loops but scale poorly due to context switching and memory overhead."),
    # cgroups confounders
    ("x_cg_seccomp",    "seccomp (Secure Computing Mode) restricts which system calls a process can make. The strict mode allows only read, write, _exit, and sigreturn. BPF-based seccomp filters evaluate each system call number and arguments against a BPF program. Container runtimes apply default seccomp profiles to reduce kernel attack surface."),
    ("x_cg_namespaces", "Linux namespaces isolate global resources between processes: PID, network, mount, UTS, IPC, user, and cgroup namespaces. Containers use all seven namespace types to create isolated environments. User namespaces map container UID 0 to an unprivileged host UID, enabling rootless containers."),
    ("x_cg_systemd",    "systemd uses cgroups to manage service resource limits and track process membership. Each service runs in its own cgroup slice; MemoryLimit, CPUQuota, and IOWeight directives map to cgroup parameters. systemd-cgls visualises the cgroup tree; systemd-cgtop shows real-time resource usage per service."),
    ("x_cg_ebpf2",      "eBPF programs can attach to cgroup hooks to enforce network and security policies per container. The cgroup/sock_ops and cgroup/skb eBPF program types intercept socket operations and network packets for processes in specific cgroups. Cilium uses this for per-pod network policies without iptables rules."),
    ("x_cg_oom",        "The Linux OOM killer terminates processes when the kernel cannot satisfy memory allocation requests. Cgroup memory limits trigger an OOM condition only within the cgroup, killing the highest oom_score_adj process in that cgroup. Container runtimes set memory limits that trigger OOM kill before the node runs out of memory."),
    # Virtual memory confounders
    ("x_vm_tlb",        "The Translation Lookaside Buffer (TLB) is a cache of recent virtual-to-physical address translations maintained by the MMU. TLB misses trigger a page table walk, which takes 100-1000 clock cycles. Context switches flush the TLB (or use ASIDs to avoid flushing). Large pages reduce TLB misses by covering more address space per entry."),
    ("x_vm_swap",       "The Linux swap subsystem evicts infrequently accessed anonymous pages to a swap partition or file to reclaim physical RAM. The swappiness parameter (0-200) controls the kernel's preference for swapping versus reclaiming page cache. zswap compresses swapped pages in RAM before writing to disk, reducing swap I/O."),
    ("x_vm_buddy",      "The Linux buddy allocator manages physical memory in power-of-two-sized blocks called buddies. Allocation finds the smallest sufficient buddy block, splitting larger blocks as needed. Deallocation merges adjacent buddies of the same size into larger blocks. This prevents external fragmentation for large allocations."),
    ("x_vm_segfault",   "A segmentation fault occurs when a process accesses memory outside its valid virtual address space, such as dereferencing a null or dangling pointer. The MMU raises a protection fault; the kernel delivers SIGSEGV. Address Space Layout Randomisation (ASLR) randomises the base addresses of mappings to complicate exploitation."),
    ("x_vm_overcommit", "Linux memory overcommitment allows the kernel to allocate more virtual memory than physical RAM plus swap. The heuristic mode (overcommit_memory=0) allows reasonable overcommitment; the nevercommit mode (2) limits virtual memory to physical RAM plus swap. Overcommitment enables efficient use of sparse virtual address spaces."),
    # eBPF confounders
    ("x_ebpf_perf",     "Linux perf is a performance analysis tool that attaches to hardware performance counters, software events, tracepoints, and kprobes. Unlike eBPF programs which run in the kernel, perf records events to a ring buffer and processes them in user space. perf stat, record, and report are the primary interfaces."),
    ("x_ebpf_dtrace",   "DTrace originated on Solaris as a dynamic tracing framework for production systems. Linux DTrace ports exist, but eBPF is the native Linux equivalent. Both attach probes to kernel and user-space functions dynamically. eBPF's verifier ensures safety; DTrace uses a similar restriction to prevent infinite loops."),
    ("x_ebpf_xdp",      "XDP (eXpress Data Path) eBPF programs attach to network drivers before the kernel networking stack processes packets. XDP_DROP discards packets at line rate; XDP_REDIRECT sends packets to another interface. XDP enables hardware-accelerated DoS mitigation and kernel bypass load balancing without user-space overhead."),
    ("x_ebpf_btf",      "BTF (BPF Type Format) encodes kernel type information in a compact binary format, enabling eBPF programs to be compiled against a kernel-independent type system. CO-RE (Compile Once, Run Everywhere) uses BTF to relocate field offsets at load time, allowing a single eBPF binary to run across different kernel versions."),
    ("x_ebpf_cilium",   "Cilium uses eBPF to implement Kubernetes network policies, load balancing, and observability without iptables. kube-proxy replacement uses eBPF maps for O(1) service translation. Cilium's Hubble provides flow-level visibility using eBPF perf events and ring buffers to record network traffic metadata."),
    # Spark DAG confounders
    ("x_spark_flink",   "Apache Flink is a stream processing framework that treats batch processing as a special case of streaming. Unlike Spark's micro-batch streaming model, Flink processes events with true event-time semantics and low latency. Flink's DAG of stateful operators runs continuously; checkpoints to durable storage enable fault recovery."),
    ("x_spark_dask",    "Dask is a parallel computing library for Python that scales NumPy, pandas, and scikit-learn workflows. Like Spark, it builds a task graph (DAG) that is executed lazily. Dask operates on in-memory partitioned data rather than Spark's RDD/DataFrame abstractions, making it simpler for Python data science workflows."),
    ("x_spark_shuffle", "Spark shuffle transfers data between stages when a wide transformation (groupBy, reduceByKey, join) requires all records with the same key to be co-located on the same executor. Shuffle writes intermediate data to disk and network-transfers it, making shuffle the dominant cost in large Spark jobs."),
    ("x_spark_catalyst", "The Spark Catalyst optimiser transforms an unresolved logical plan into a physical execution plan through analysis, logical optimisation (predicate pushdown, constant folding), physical planning, and code generation via Tungsten. The optimiser automatically rewrites queries for better performance without user intervention."),
    ("x_spark_stream",  "Spark Structured Streaming treats a real-time data stream as an unbounded table. Queries execute incrementally as new data arrives. Triggers control when processing occurs: default (process as fast as possible), fixed interval (mini-batch), or once (process all pending data once). Checkpointing enables exactly-once semantics."),
    # Kafka confounders
    ("x_kafka_rabbitmq","RabbitMQ is a message broker implementing AMQP. Unlike Kafka's log-based storage, RabbitMQ routes messages through exchanges to queues and deletes them after acknowledgement. RabbitMQ supports complex routing patterns (topic, fanout, direct) and message priorities; Kafka retains messages by time or size regardless of consumption."),
    ("x_kafka_ksql",    "ksqlDB is a streaming SQL engine for Kafka. It allows filtering, aggregating, and joining Kafka streams and tables using SQL syntax without writing Java code. ksqlDB materialises results as Kafka topics or queryable state stores. Stream-table joins enable enriching event streams with reference data."),
    ("x_kafka_connect", "Kafka Connect is a framework for streaming data between Kafka and external systems using source and sink connectors. Connectors run as distributed workers; tasks read from or write to the external system in parallel. Debezium provides CDC (Change Data Capture) source connectors for databases."),
    ("x_kafka_avro",    "Apache Avro is a binary serialisation format used with Kafka's Schema Registry. Schemas are stored centrally and referenced by ID in message headers. Avro enables schema evolution with backward and forward compatibility rules. Confluent's Schema Registry enforces compatibility checks on schema registration."),
    ("x_kafka_lag",     "Consumer lag measures how far behind a Kafka consumer group is from the latest offset in each partition. High lag indicates the consumer cannot keep up with producer throughput. Lag can be monitored with kafka-consumer-groups.sh or the __consumer_offsets internal topic. Increasing partition count enables more parallel consumers."),
    # Parquet confounders
    ("x_parq_orc",      "ORC (Optimised Row Columnar) is a columnar storage format developed for Hive. Like Parquet, ORC stores data column by column and supports predicate pushdown via column statistics. ORC uses ZLIB or Snappy compression; Parquet supports SNAPPY, GZIP, BROTLI, and ZSTD. ORC is more common in the Hadoop ecosystem."),
    ("x_parq_arrow",    "Apache Arrow defines an in-memory columnar data format for zero-copy data sharing between systems. Unlike Parquet (a disk format), Arrow is designed for CPU-efficient analytics with SIMD-friendly memory layouts. Arrow IPC allows sharing Arrow buffers between processes; the Flight protocol streams Arrow over gRPC."),
    ("x_parq_delta",    "Delta Lake is an open table format built on Parquet files with an ACID transaction log. The _delta_log directory records all commits as JSON files, enabling time travel, concurrent writes with optimistic concurrency, and schema enforcement. Delta Lake adds MERGE, UPDATE, and DELETE operations to the Parquet-only append model."),
    ("x_parq_compress", "Parquet compression codecs differ in speed and ratio: SNAPPY prioritises decompression speed (useful for Spark); ZSTD provides better compression ratio with tunable levels; GZIP offers high compression at lower speed. Dictionary encoding for low-cardinality columns often achieves better compression than general-purpose codecs."),
    ("x_parq_hive",     "Hive partitioning organises data in directories by partition column values (e.g., year=2024/month=01/). Partition pruning skips directories that don't match query predicates. Hive metastore stores table and partition metadata; Spark and Trino use it for partition discovery without scanning the filesystem."),
    # dbt confounders
    ("x_dbt_airflow",   "Apache Airflow orchestrates data pipelines as DAGs of tasks with dependencies. Unlike dbt, which transforms data within the warehouse using SQL, Airflow coordinates multi-system workflows including data extraction, loading, and transformation steps. dbt is often triggered by Airflow as a step in a broader pipeline."),
    ("x_dbt_great",     "Great Expectations is a data quality framework that validates data using customisable expectations (e.g., column values not null, unique, between range). Unlike dbt tests (which run SQL checks), Great Expectations works across multiple data stores and generates HTML validation reports for data documentation."),
    ("x_dbt_snapshot",  "dbt snapshots track slowly changing dimension (SCD Type 2) changes by appending historical rows with dbt_valid_from and dbt_valid_to timestamps. When source records change, the snapshot updates the previous row's valid_to and inserts a new row. This enables point-in-time queries on historical data states."),
    ("x_dbt_macro",     "dbt macros are Jinja2 templates that generate SQL dynamically. The dbt_utils package provides generic macros for common patterns: date_trunc, surrogate_key, pivot, and union_relations. Custom macros abstract repeated SQL patterns into reusable functions called with Jinja2 syntax in model SQL files."),
    ("x_dbt_semantic",  "MetricFlow is the metric layer integrated into dbt, defining metrics as reusable semantic objects separate from models. Metrics specify aggregation type, dimensions, and time granularity. The semantic layer enables consistent metric definitions across BI tools by querying the warehouse through a unified interface."),
    # Iceberg confounders
    ("x_ice_hudi",      "Apache Hudi is an open table format focused on incremental data processing. Like Iceberg, it supports ACID transactions and time travel. Hudi offers two table types: Copy-on-Write (rewrites files on update) and Merge-on-Read (appends delta files, merging on read). Hudi is optimised for streaming ingestion into lakes."),
    ("x_ice_delta2",    "Delta Lake (by Databricks) competes directly with Iceberg. Both provide ACID transactions on Parquet files with time travel and schema evolution. Delta Lake uses a JSON-based transaction log; Iceberg uses a hierarchical metadata tree. The Uniform format (dbt) allows a single table to expose both Delta and Iceberg APIs."),
    ("x_ice_catalog",   "The Iceberg REST Catalog API standardises how compute engines discover and manage Iceberg tables. Polaris Catalog (Apache Iceberg's reference implementation), Nessie (Git-for-data branching), and Tabular provide REST catalog implementations. A catalog stores the current metadata pointer for each table."),
    ("x_ice_z_order",   "Z-ordering (Hilbert sorting) clusters rows in Parquet files by multiple columns simultaneously, enabling better data skipping than single-column sorting. Iceberg's data-skipping uses min/max statistics in manifest files to skip entire files. Z-ordering improves multi-dimensional filter selectivity at the cost of compaction time."),
    ("x_ice_format_v3", "Iceberg table format v3 adds row-level deletes using deletion vectors (bitset tracking deleted rows per file), multi-valued collation keys for custom sorting, and nanosecond timestamp types. Deletion vectors avoid rewriting files for delete operations, reducing write amplification compared to copy-on-write approaches."),
]

# ---------------------------------------------------------------------------
# 40 queries with ground-truth doc IDs
# ---------------------------------------------------------------------------

QUERIES: list[tuple[str, str, str]] = [
    # (query_id, query_text, correct_chunk_id)
    # -- Python --
    ("q01", "interpreted high-level language readability indentation paradigms",               "c_py_interp"),
    ("q02", "asyncio coroutines event loop async await concurrent Python",                     "c_py_async"),
    # -- Go --
    ("q03", "goroutines channels lightweight concurrency Go runtime",                          "c_go_concur"),
    # -- Rust --
    ("q04", "ownership borrowing memory safety no garbage collector Rust compile time",        "c_rust_own"),
    # -- TypeScript --
    ("q05", "structural type system TypeScript generics compile JavaScript",                   "c_ts_types"),
    # -- ML --
    ("q06", "chain rule backward pass gradient weights neural network loss",                   "c_backprop"),
    ("q07", "self-attention multi-head positional encoding transformer architecture",          "c_transformer"),
    ("q08", "maximum margin hyperplane kernel trick non-linear classification",                "c_svm"),
    ("q09", "k centroids iterative assignment cluster mean minimise squared distance",         "c_kmeans"),
    ("q10", "gradient boosted trees L1 L2 regularisation approximate split finding",          "c_xgboost"),
    # -- Databases --
    ("q11", "MVCC snapshot readers writers PostgreSQL dead tuples VACUUM",                     "c_pg_mvcc"),
    ("q12", "aggregation pipeline stages match group project lookup MongoDB",                  "c_mongo_agg"),
    ("q13", "in-memory data structures strings hashes sets sorted sets pub sub Redis",         "c_redis_ds"),
    ("q14", "wide column store consistent hashing gossip protocol peer-to-peer Cassandra",     "c_cassandra"),
    ("q15", "distributed RESTful search inverted index shards replicas Lucene Elasticsearch",  "c_elastic"),
    # -- Security --
    ("q16", "AES block cipher SubBytes ShiftRows MixColumns AddRoundKey symmetric rounds",     "c_aes"),
    ("q17", "prime factorisation public key private key RSA asymmetric",                       "c_rsa"),
    ("q18", "TLS handshake certificate ECDHE forward secrecy cipher suite",                   "c_tls"),
    ("q19", "inject script CSP output encoding stored reflected XSS",                         "c_xss"),
    ("q20", "parameterised queries prepared statements SQL injection",                         "c_sqli"),
    # -- Cloud --
    ("q21", "Kubernetes scheduler filter score pod node affinity CPU memory",                  "c_k8s_sched"),
    ("q22", "Docker image union filesystem layers copy-on-write writable container",           "c_docker_layer"),
    ("q23", "Terraform HCL plan apply state infrastructure as code",                          "c_terraform"),
    ("q24", "Prometheus scrape metrics PromQL rate histogram_quantile alerting",               "c_prometheus"),
    ("q25", "nginx event-driven non-blocking reverse proxy upstream worker",                   "c_nginx"),
    # -- Networking --
    ("q26", "TCP congestion control slow start window cubic packet loss BBR",                  "c_tcp_flow"),
    ("q27", "DNS recursive resolver root nameserver TTL DNSSEC authoritative",                 "c_dns_resolv"),
    ("q28", "HTTP/2 multiplexed streams single connection header compression HPACK",           "c_http2"),
    ("q29", "BGP autonomous systems path vector AS path local preference eBGP iBGP",          "c_bgp"),
    ("q30", "QUIC UDP encrypted multiplexed streams connection migration 0-RTT",               "c_quic"),
    # -- Algorithms --
    ("q31", "shortest path weighted graph non-negative priority queue Dijkstra",               "c_dijkstra"),
    ("q32", "probabilistic set membership false positive bit array hash functions Bloom",      "c_bloom"),
    ("q33", "B-tree balanced branching factor leaf sorted database index I/O",                 "c_btree"),
    ("q34", "memtable SSTable compaction write-heavy LSM log-structured merge",                "c_lsm"),
    ("q35", "virtual ring hash node distribution minimal redistribution consistent hashing",   "c_consistent_hash"),
    # -- Web --
    ("q36", "React Fiber incremental reconciler concurrent startTransition commit phase",      "c_react_fiber"),
    ("q37", "GraphQL client specifies fields resolver single endpoint typed schema",           "c_graphql"),
    ("q38", "CORS preflight Access-Control-Allow-Origin cross-origin browser",                 "c_cors"),
    ("q39", "SSR server-side rendering hydration First Contentful Paint SEO",                  "c_ssr"),
    ("q40", "WebSocket full-duplex persistent connection upgrade real-time",                   "c_websockets"),
]


def build_corpus() -> list[tuple[str, str]]:
    """Return all (chunk_id, content) pairs in fixed order."""
    return list(CORE_DOCS) + list(CONFOUNDERS)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _mrr(results: list[str], correct: str) -> float:
    for rank, r in enumerate(results, 1):
        if r == correct:
            return 1.0 / rank
    return 0.0


def _hit_at_k(results: list[str], correct: str, k: int) -> int:
    return int(correct in results[:k])


def run_measurement(
    retriever,
    cross_encoder_model: str | None,
    *,
    top_k: int = 10,
) -> dict:
    """Run all 40 queries across 4 modes; return per-query metric dicts."""
    from app.core.retriever import (
        _reciprocal_rank_fusion,
        _score_rerank,
    )

    per_query: dict[str, dict] = {}

    for qid, query, correct_id in QUERIES:
        vec = retriever._vector_search(query, top_k * 2)
        fts = retriever._fts5_search(query, top_k * 2)
        rrf = _reciprocal_rank_fusion(list(vec), list(fts), k=60)
        composite = _score_rerank(query, list(rrf))

        # Cross-encoder: only if model provided (measurement-only, not a runtime dep)
        if cross_encoder_model:
            try:
                from sentence_transformers.cross_encoder import CrossEncoder
                if not hasattr(run_measurement, "_ce"):
                    run_measurement._ce = CrossEncoder(cross_encoder_model, device="cpu", max_length=512)
                ce = run_measurement._ce
                pairs = [(query, chunk.content[:512]) for chunk in rrf]
                scores = ce.predict(pairs, show_progress_bar=False)
                scored = [(chunk, float(score)) for chunk, score in zip(rrf, scores)]
                ce_reranked = [c for c, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
            except Exception as exc:
                print(f"  [WARN] CE rerank failed for {qid}: {exc}", file=sys.stderr)
                ce_reranked = list(rrf)
        else:
            ce_reranked = None

        per_query[qid] = {
            "correct":   correct_id,
            "vec":       [r.chunk_id for r in vec],
            "rrf":       [r.chunk_id for r in rrf],
            "composite": [r.chunk_id for r in composite],
            "ce":        [r.chunk_id for r in ce_reranked] if ce_reranked else None,
        }

    return per_query


def compute_metrics(per_query: dict, mode: str, k: int = 5) -> dict[str, float]:
    """Compute Recall@k, P@5, P@1, MRR for a given mode."""
    if mode == "ce" and all(v["ce"] is None for v in per_query.values()):
        return {}

    hits_k = p1 = mrr_sum = 0
    p5_hits = 0
    total = len(per_query)

    for d in per_query.values():
        results = d[mode]
        if results is None:
            continue
        correct = d["correct"]
        hits_k += _hit_at_k(results, correct, k)
        p1     += _hit_at_k(results, correct, 1)
        p5_hits += _hit_at_k(results, correct, 5)
        mrr_sum += _mrr(results, correct)

    return {
        f"Recall@{k}": hits_k / total,
        "P@5":         p5_hits / total,
        "P@1":         p1 / total,
        "MRR":         mrr_sum / total,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def main():
    import chromadb
    from app.database import SafeDB
    from app.core.retriever import Retriever

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="", help="Cross-encoder model ID (empty=skip CE mode)")
    parser.add_argument("--top-k", type=int, default=10, help="Candidates per search arm (default 10)")
    args = parser.parse_args()

    corpus = build_corpus()
    print(f"Corpus: {len(corpus)} documents ({len(CORE_DOCS)} core + {len(CONFOUNDERS)} confounders)")
    print(f"Queries: {len(QUERIES)}")
    print()

    # Setup temp DB + ChromaDB
    tmp = tempfile.mkdtemp()
    db_path = pathlib.Path(tmp) / "eval.db"
    db = SafeDB(str(db_path))
    db.init_schema()

    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.create_collection("eval_300", metadata={"hnsw:space": "cosine"})

    print("Seeding FTS5 + ChromaDB ... ", end="", flush=True)
    for cid, content in corpus:
        db.execute(
            "INSERT INTO chunks_fts (chunk_id, document_id, content) VALUES (?, ?, ?)",
            (cid, "doc_" + cid[:8], content),
        )
    collection.add(
        ids=[c[0] for c in corpus],
        documents=[c[1] for c in corpus],
        metadatas=[{"document_id": "doc_" + c[0][:8], "source": "eval", "title": c[0]} for c in corpus],
    )
    print("done.")
    print()

    retriever = Retriever(db=db, chroma_collection=collection)

    ce_model = args.model or None
    if ce_model:
        print(f"Cross-encoder: {ce_model} (loading on first query)")
    else:
        print("Cross-encoder: skipped (pass --model <id> to enable)")
    print()

    print("Running 40 queries across modes ... ", end="", flush=True)
    per_query = run_measurement(retriever, ce_model, top_k=args.top_k)
    print("done.")
    print()

    # Build table
    modes = [
        ("vec",       "Pure vector"),
        ("rrf",       "Hybrid RRF (no rerank)"),
        ("composite", "Hybrid + composite"),
    ]
    if ce_model:
        modes.append(("ce", f"Hybrid + cross-encoder"))

    headers = ["Mode", "Recall@5", "P@5", "P@1", "MRR"]
    rows = []
    for mode_key, mode_label in modes:
        m = compute_metrics(per_query, mode_key, k=5)
        if not m:
            continue
        rows.append((mode_label, m["Recall@5"], m["P@5"], m["P@1"], m["MRR"]))

    col_w = [max(len(h), max(len(r[0]) for r in rows)) for h in headers[:1]] + [10] * (len(headers) - 1)
    col_w[0] = max(col_w[0], max(len(r[0]) for r in rows))

    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    hdr = "| " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers)) + " |"
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        cells = [r[0].ljust(col_w[0])] + [f"{v:.1%}".rjust(col_w[i+1]) for i, v in enumerate(r[1:])]
        print("| " + " | ".join(cells) + " |")
    print(sep)
    print()

    if ce_model and len(rows) >= 4:
        comp_r5   = rows[2][1]
        ce_r5     = rows[3][1]
        comp_mrr  = rows[2][4]
        ce_mrr    = rows[3][4]
        comp_p1   = rows[2][3]
        ce_p1     = rows[3][3]
        beats_any = (
            (ce_r5   - comp_r5)  * 100 >= 3 or
            (ce_mrr  - comp_mrr) * 100 >= 3 or
            (ce_p1   - comp_p1)  * 100 >= 3
        )
        print(f"Cross-encoder vs composite: Recall@5 {(ce_r5-comp_r5)*100:+.1f}pp, "
              f"P@1 {(ce_p1-comp_p1)*100:+.1f}pp, MRR {(ce_mrr-comp_mrr)*100:+.1f}pp")
        if beats_any:
            print("Decision: cross-encoder beats composite by >=3 pp on at least one metric -> KEEP as opt-in")
        else:
            print("Decision: cross-encoder does NOT beat composite by >=3 pp on any metric -> DELETE path")

    return rows


if __name__ == "__main__":
    rows = asyncio.run(main())
