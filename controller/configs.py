from dataclasses import dataclass, asdict

@dataclass
class TrainConfig:
    seed: int = 0
    log_dir: str = 'log'
    state_dim: int = 30
    rnn_hidden_dim: int = 200
    rnn_input_dim: int = 200
    min_std: float = 1e-2
    num_episodes: int = 100
    batch_size: int = 50
    num_epochs: int = 1024
    chunk_length: int = 50
    test_size: float = 0.2
    lr: float = 1e-3
    eps: float = 1e-5
    clip_grad_norm: int = 1000
    free_nats: int = 0
    kl_beta: float = 1
    
    dict = asdict