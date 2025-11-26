#!/usr/bin/env python3
"""
Simple OpenAI-compatible API server using transformers directly.

This is a minimal reference implementation that provides a lightweight local
inference server. It is intentionally kept simple - just wraps a transformers
model with OpenAI API endpoints.

Design Philosophy:
- SIMPLE: Just model wrapper, no special formatting logic
- GENERIC: Works with any transformers model
- TRANSPARENT: Easy to understand and modify (~100 lines)

All Harmony formatting, prompt engineering, and response parsing is handled
by the GENERATOR (generate_best_of_n.py), not the server.

Useful for:
- Testing Best-of-N generation without API costs
- QA/validation of Best-of-N changes
- Working with NEXUS-modified models
- Offline development and debugging
- Any transformers model (GPT-OSS, Llama, etc.)

Features:
- OpenAI-compatible /v1/chat/completions endpoint
- Support for n>1 batch generation
- Simple message concatenation (no special formatting)
- Health check endpoint

Limitations:
- Sequential generation (slower than vLLM/TGI)
- Single GPU only (no tensor parallelism)
- No advanced optimizations (PagedAttention, etc.)
- Not production-grade (no auth, rate limiting, etc.)

Usage:
    # Start server
    python local_server.py --model /path/to/your-model --port 8000

    # In another terminal
    export OPENAI_BASE_URL=http://localhost:8000/v1
    export OPENAI_API_KEY=dummy

    python generate_best_of_n.py --model your-model --config experiments/baseline.yaml

Requirements:
    pip install flask transformers torch
"""

import argparse
import time
import uuid
import logging
import threading
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, GenerationConfig
from transformers.distributed import DistributedConfig
import torch
torch.set_float32_matmul_precision('high')
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("LocalServer")

app = Flask(__name__)
app.logger.setLevel(logging.WARNING)  # Reduce Flask noise

# Global model, tokenizer, and GPU lock
model = None
tokenizer = None
model_name = None
gpu_lock = threading.Lock()  # Serialize GPU access for thread safety


def load_model(model_path: str):
    """Load any transformers-compatible model."""
    global model, tokenizer, model_name

    logger.info(f"Loading model from {model_path}...")
    model_name = model_path.split('/')[-1]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model (MXFP4 quantization auto-detected from checkpoint)
        # GPT-OSS NEXUS models have MXFP4 routed experts + BF16 shared expert
        # Following NEXUS model_utils.py pattern: simple load with device_map
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="auto",
            trust_remote_code=True,
            device_map="cuda:1",
        )
        model.eval()
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
        logger.info(f"  Device: {next(model.parameters()).device}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """
    OpenAI-compatible chat completions endpoint.

    Simple implementation: concatenates messages and generates.
    The CLIENT (generate_best_of_n.py) handles all special formatting (Harmony, etc.).
    """
    start_time = time.time()

    try:
        data = request.json

        # Extract parameters
        messages = data.get("messages", [])
        n = data.get("n", 1)
        temperature = data.get("temperature", 1.0)
        max_tokens = data.get("max_tokens", 100000)  # Safe default, not 131K!

        # Cap max_tokens to prevent OOM
        max_tokens = min(max_tokens, 100000)

        logger.debug(f"Request: n={n}, temp={temperature}, max_tokens={max_tokens}")

        # Efficient prompt construction with list join (O(N) not O(N²))
        prompt_parts = []
        for msg in messages:
            content = msg.get("content", "")
            if content:  # Only add non-empty content
                prompt_parts.append(str(content))
        prompt_parts.append("<|start|>assistant")
        prompt = "\n\n".join(prompt_parts)

        # Edge case: empty prompt
        if not prompt or not prompt.strip():
            return jsonify({"error": {"message": "Empty prompt", "code": "invalid_request"}}), 400

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model.device)
        prompt_tokens = inputs.input_ids.shape[1]

        # Check context window overflow
        if prompt_tokens + max_tokens > getattr(model.config, 'max_position_embeddings', 100000):
            return jsonify({"error": {"message": "Prompt + max_tokens exceeds context window", "code": "context_overflow"}}), 400

        # MICRO-BATCHING: Generate N completions in physical batches
        # Physical limit depends on GPU memory:
        #   - 32 for A100 (80GB)
        #   - 16 for A6000/A40 (48GB)
        #   - 8 for RTX 4090 (24GB)
        PHYSICAL_BATCH_SIZE = 24

        choices = []
        total_completion_tokens = 0



        try:
            with gpu_lock:  # Serialize GPU access - thread-safe
                # Process in micro-batches to avoid OOM on large n
                for batch_start in range(0, n, PHYSICAL_BATCH_SIZE):
                    current_batch_size = min(PHYSICAL_BATCH_SIZE, n - batch_start)

                    # Log micro-batch progress for large n
                    if n > PHYSICAL_BATCH_SIZE:
                        batch_num = batch_start // PHYSICAL_BATCH_SIZE + 1
                        total_batches = (n + PHYSICAL_BATCH_SIZE - 1) // PHYSICAL_BATCH_SIZE
                        logger.info(f"Micro-batch {batch_num}/{total_batches}: generating candidates {batch_start}-{batch_start + current_batch_size - 1}...")

                    with torch.no_grad():

                        generation_config = GenerationConfig(
                                max_new_tokens=max_tokens,
                                temperature=temperature,
                                do_sample=temperature > 0,
                                num_return_sequences=n,
                                pad_token_id=tokenizer.pad_token_id,
                                use_cache=True,
                                max_batch_tokens=4096,  # Process prompt in chunks if too large
                        )
                        logger.info(f"Generation config: {generation_config}, current_batch_size={current_batch_size}, temp={temperature}, max_tokens={max_tokens}")
                        logger.info(f"Length of input ids: {inputs.input_ids.shape[1]}")
                        batch_outputs = model.generate(
                            **inputs,
                            generation_config=generation_config,
                        )

                    # Decode immediately to free GPU memory
                    for batch_idx in range(current_batch_size):
                        generated_ids = batch_outputs[batch_idx][prompt_tokens:]
                        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
                        completion_tokens = len(generated_ids)
                        total_completion_tokens += completion_tokens
                        logger.info(f"Generated text: {generated_text}")
                        choices.append({
                            "index": batch_start + batch_idx,  # Global index across batches
                            "message": {
                                "role": "assistant",
                                "content": generated_text
                            },
                            "finish_reason": "stop"
                        })

                    # Free GPU memory before next batch
                    del batch_outputs

                    # Clear CUDA cache between batches
                    if batch_start + PHYSICAL_BATCH_SIZE < n:  # More batches coming
                        torch.cuda.empty_cache()

                        # Extra safety: Force garbage collection
                        import gc
                        gc.collect()

        except torch.cuda.OutOfMemoryError:
            # Explicit OOM handling
            torch.cuda.empty_cache()
            logger.error("CUDA OOM detected - try reducing max_tokens or n")
            return jsonify({
                "error": {
                    "code": "oom",
                    "message": "GPU Out of Memory. Try smaller max_tokens or fewer candidates (n)."
                }
            }), 503
        except Exception as gen_error:
            logger.error(f"Generation failed: {gen_error}")
            return jsonify({"error": {"code": "generation_error", "message": str(gen_error)}}), 500

        elapsed = time.time() - start_time

        # Calculate performance metrics
        tokens_per_second = total_completion_tokens / elapsed if elapsed > 0 else 0
        total_tokens = prompt_tokens + total_completion_tokens

        logger.info(f"=" * 80)
        logger.info(f"Request completed:")
        logger.info(f"  Completions: {n}")
        logger.info(f"  Time: {elapsed:.2f}s (avg: {elapsed/n:.2f}s per completion)")
        logger.info(f"  Prompt tokens: {prompt_tokens}")
        logger.info(f"  Completion tokens: {total_completion_tokens} (avg: {total_completion_tokens/n:.0f} per completion)")
        logger.info(f"  Total tokens: {total_tokens}")
        logger.info(f"  Throughput: {tokens_per_second:.1f} tokens/sec")
        logger.info(f"  Generation speed: {total_completion_tokens / elapsed:.1f} completion tokens/sec")
        if n > 1:
            logger.info(f"  Batch efficiency: {n} sequences generated in parallel")
        logger.info(f"=" * 80)

        # Return OpenAI-compatible response
        return jsonify({
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": prompt_tokens + total_completion_tokens
            }
        })

    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": "internal_error"
            }
        }), 500


@app.route("/v1/responses", methods=["POST"])
def responses():
    """
    OpenAI Responses API endpoint.

    Uses instructions + input format (newer OpenAI API).
    Always returns a single response (n=1).
    """
    start_time = time.time()

    try:
        data = request.json

        # Extract Responses API parameters
        instructions = data.get("instructions", "You are a helpful assistant.")
        user_input = data.get("input", "")
        temperature = data.get("temperature", 1.0)
        max_output_tokens = data.get("max_output_tokens", 4096)

        # Build prompt: instructions + input
        prompt = f"{instructions}\n\n{user_input}"

        # Edge case: empty prompt
        if not prompt or not prompt.strip():
            return jsonify({"error": {"message": "Empty prompt", "code": "invalid_request"}}), 400

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model.device)
        prompt_tokens = inputs.input_ids.shape[1]

        # Check context window
        if prompt_tokens + max_output_tokens > getattr(model.config, 'max_position_embeddings', 100000):
            return jsonify({"error": {"message": "Input exceeds context window", "code": "context_overflow"}}), 400

        # Generate single response (Responses API doesn't support n>1)
        try:
            with gpu_lock:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_output_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True,
                    )

                generated_ids = outputs[0][prompt_tokens:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                completion_tokens = len(generated_ids)

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error("CUDA OOM during generation")
            return jsonify({
                "error": {
                    "code": "oom",
                    "message": "GPU Out of Memory. Try smaller max_output_tokens."
                }
            }), 503
        except Exception as gen_error:
            logger.error(f"Generation failed: {gen_error}")
            return jsonify({"error": {"code": "generation_error", "message": str(gen_error)}}), 500

        elapsed = time.time() - start_time

        # Log metrics
        logger.info(f"Response completed in {elapsed:.2f}s")
        logger.info(f"  Tokens: {prompt_tokens} prompt + {completion_tokens} completion")
        logger.info(f"  Throughput: {completion_tokens / elapsed:.1f} tokens/sec")

        # Return OpenAI Responses API format
        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        return jsonify({
            "id": response_id,
            "object": "response",
            "created": int(time.time()),
            "model": model_name,
            "output_items": [
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": generated_text.strip()
                    }
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        })

    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": "internal_error"
            }
        }), 500


@app.route("/v1/models", methods=["GET"])
def list_models():
    """List available models."""
    return jsonify({
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local"
        }]
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model_loaded": model is not None})


def main():
    parser = argparse.ArgumentParser(
        description="Simple OpenAI-compatible API server for local inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with NEXUS model
  python local_server.py --model /path/to/gpt-oss-120b-nexus --port 8000

  # Use with Best-of-N
  export OPENAI_BASE_URL=http://localhost:8000/v1
  export OPENAI_API_KEY=dummy
  python generate_best_of_n.py --model gpt-oss-120b-nexus --config experiments/baseline.yaml

  # Test health
  curl http://localhost:8000/health
        """
    )
    parser.add_argument("--model", required=True, help="Path to model (NEXUS or any transformers model)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    # Load model
    load_model(args.model)

    # Print startup info
    logger.info("")
    logger.info("=" * 80)
    logger.info("Local OpenAI-Compatible Server Ready")
    logger.info("=" * 80)
    logger.info(f"Server URL: http://{args.host}:{args.port}")
    logger.info(f"Endpoints:")
    logger.info(f"  POST /v1/chat/completions (supports n>1)")
    logger.info(f"  POST /v1/responses (OpenAI Responses API)")
    logger.info(f"  GET  /v1/models")
    logger.info(f"  GET  /health")
    logger.info("")
    logger.info("To use with Best-of-N:")
    logger.info(f"  export OPENAI_BASE_URL=http://localhost:{args.port}/v1")
    logger.info(f"  export OPENAI_API_KEY=dummy")
    logger.info(f"  python generate_best_of_n.py --model {model_name} ...")
    logger.info("")
    logger.info("Note: Auto-detects API format (chat completions vs responses)")
    logger.info("")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 80)
    logger.info("")

    # Run server (threaded for concurrent requests)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
