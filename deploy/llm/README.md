# Answer model and query embeddings on a GPU machine

The public host (no GPU) serves the search. For answers and fast query encoding it talks to a GPU machine
through a reverse SSH tunnel that the GPU machine opens itself, so nothing on the GPU machine listens to the
outside.

```
browser -> Caddy -> API on the host -> 127.0.0.1:18100 -+  reverse tunnel   +-> vLLM (answer model)
                                    -> 127.0.0.1:18101 -+  (opened from the  +-> embed_service.py (Jina v4 queries)
                                                           GPU machine)
```

| File | What it does |
|---|---|
| `run_vllm.sh` | OpenAI-compatible vLLM server on `127.0.0.1:18100`, API key required. `MODEL` selects the model (default `Qwen/Qwen3.8-27B-FP8`). No `--quantization` flag: vLLM reads it from the checkpoint. |
| `embed_service.py`, `run_embed.sh` | `jina-embeddings-v4` for queries on `127.0.0.1:18101`, API key required. |
| `tunnel.sh` | Restart loop around `ssh -N -R`, forwards both ports to the host. |
| `start_all.sh` | Starts whatever of the three is not running, each detached with a pidfile. |

Setup in short: create a key pair for the tunnel, add its public key on the host with
`restrict,port-forwarding,command="/bin/false",permitlisten="127.0.0.1:18100",permitlisten="127.0.0.1:18101"`
in front, put one shared secret in a file (default `~/.config/secrets/plenarsuche_llm.key`), set
`PARLIAMENT_LLM_URL`, `PARLIAMENT_LLM_KEY` and `PARLIAMENT_EMBED_URL` in the host's environment file, restart
the API.

The answer feature is a convenience on top of the search. If the GPU machine or the tunnel is down, the
interface hides the switch and searches fall back to the Jina API.
