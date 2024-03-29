# Cerebro
Resource-Efficient and Reproducible Model Selection on Deep Learning Systems

### Overview
Artificial Neural Networks (ANNs) are revolutionizing many machine learning (ML) applications. Their success at major Web companies has created excitement among many enterprises and domain scientists to try ANNs for their applications. But training ANNs is a notoriously painful empirical process, since accuracy is tied to the ANN architecture and hyper-parameter settings. The common practice to choose these settings is to empirically compare as many training configurations as feasible for the application. This process is called model selection, and it is unavoidable because it is how one controls underfitting vs overfitting. Model selection is a major bottleneck for adoption of ANNs among enterprises and domain scientists due to both the time spent and resource costs.

In this project, we propose a new system for ANN model selection that raises model selection throughput without raising resource costs. Our target setting is small clusters (say, 10s of nodes), which covers a vast majority (almost 90%) of parallel ML workloads in practice. We have 4 key system desiderata: scalability, statistical convergence efficiency, reproducibility, and system generality. To satisfy all these desiderata, we develop a novel parallel execution strategy we call model hopper parallelism (MOP).
