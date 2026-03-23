"""Fine-tuning and model alignment demo

This demo illustrates the behavioral differences that fine-tuning produces by
comparing a genuine base model against its instruction-tuned counterpart, both
loaded locally via HuggingFace transformers.  It also includes a dataset
formatter tab that shows what SFT and DPO training data actually looks like.

Tabs:
1. Model comparison   - same prompt sent to a base model and an instruction-tuned
                        model simultaneously; responses shown side by side
2. Dataset formatter  - format instruction/output pairs into Alpaca, ChatML, and
                        DPO JSON structures used for fine-tuning

Usage:
    # Models are downloaded from HuggingFace on first run (~500 MB each).
    # Set HF_HOME to control the cache directory if needed.

    python demos/finetuning/finetuning_demo.py
"""

import json
import threading
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# --- Configuration ---

# Genuine base and instruct checkpoints from HuggingFace.
# These are the actual separate model weights - not aliases of the same file.
#
# Other pairs to try (larger = better contrast, more VRAM needed):
#   "Qwen/Qwen2.5-1.5B"              + "Qwen/Qwen2.5-1.5B-Instruct"   (~3 GB)
#   "meta-llama/Llama-3.2-1B"        + "meta-llama/Llama-3.2-1B-Instruct"  (gated)

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B"
INSTRUCT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Generation settings
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7


def _best_device() -> str:
    """Return the CUDA device with the most free VRAM, or CPU."""
    if not torch.cuda.is_available():
        return "cpu"
    free_mem = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return f"cuda:{free_mem.index(max(free_mem))}"


DEVICE = _best_device()
DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32


# --- Lazy model loading ---

_models: dict = {}
_load_lock = threading.Lock()


def _load_model(model_id: str) -> None:
    """Download (if needed) and load a model + tokenizer into the in-memory cache."""
    if model_id in _models:
        return
    with _load_lock:
        if model_id in _models:  # double-checked locking
            return
        print(f"Loading {model_id} onto {DEVICE} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE)
        model.to(DEVICE)
        model.eval()
        _models[model_id] = (model, tokenizer)
        print(f"  {model_id} ready.")


# --- Demo functions ---

def compare_models(prompt: str) -> tuple[str, str]:
    """Send the same prompt to the base and instruct models and return both responses.

    The base model receives the raw prompt with no chat template - this is how a
    pre-trained checkpoint behaves before any instruction tuning: it continues the
    text as a completion.

    The instruct model receives the prompt via its chat template (system + user
    turns), which is the format it was supervised fine-tuned to respond to.

    Args:
        prompt: The user's input text.

    Returns:
        Tuple of (base_response, instruct_response).
    """
    if not prompt.strip():
        return "Please enter a prompt.", "Please enter a prompt."

    # -- Base model: raw text completion ----------------------------------
    # Tokenize the prompt directly with no special tokens or role framing.
    # The model will try to continue the text as if completing a document.
    try:
        _load_model(BASE_MODEL_ID)
        base_model, base_tok = _models[BASE_MODEL_ID]
        base_inputs = base_tok(prompt, return_tensors="pt")
        input_ids = base_inputs["input_ids"].to(DEVICE)
        attention_mask = base_inputs["attention_mask"].to(DEVICE)
        with torch.no_grad():
            out = base_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=base_tok.eos_token_id,
            )
        new_tokens = out[0, input_ids.shape[1]:]
        base_response = base_tok.decode(new_tokens, skip_special_tokens=True)
    except Exception as e:
        base_response = f"Error: {e}"

    # -- Instruct model: chat-template completion -------------------------
    # Apply the tokenizer's built-in chat template so the prompt arrives in
    # the exact format the model was fine-tuned on.
    try:
        _load_model(INSTRUCT_MODEL_ID)
        inst_model, inst_tok = _models[INSTRUCT_MODEL_ID]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt},
        ]
        text = inst_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inst_inputs = inst_tok(text, return_tensors="pt")
        input_ids = inst_inputs["input_ids"].to(DEVICE)
        attention_mask = inst_inputs["attention_mask"].to(DEVICE)
        with torch.no_grad():
            out = inst_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=inst_tok.eos_token_id,
            )
        new_tokens = out[0, input_ids.shape[1]:]
        instruct_response = inst_tok.decode(new_tokens, skip_special_tokens=True)
    except Exception as e:
        instruct_response = f"Error: {e}"

    return base_response, instruct_response


def format_sft_alpaca(instruction: str, context: str, output: str) -> str:
    """Format a training example in Alpaca JSON format.

    Args:
        instruction: The task instruction.
        context: Optional additional input or context.
        output: The ideal model response.

    Returns:
        Pretty-printed JSON string.
    """
    example: dict = {"instruction": instruction}

    if context.strip():
        example["input"] = context

    example["output"] = output
    return json.dumps(example, indent=2, ensure_ascii=False)


def format_sft_chatml(instruction: str, context: str, output: str) -> str:
    """Format a training example in ChatML format.

    Args:
        instruction: The user instruction.
        context: Optional additional context appended to the user turn.
        output: The ideal assistant response.

    Returns:
        ChatML-formatted string.
    """
    user_content = instruction

    if context.strip():
        user_content = f"{instruction}\n\n{context}"

    lines = [
        "<|im_start|>system",
        "You are a helpful assistant.<|im_end|>",
        "<|im_start|>user",
        f"{user_content}<|im_end|>",
        "<|im_start|>assistant",
        f"{output}<|im_end|>",
    ]
    return "\n".join(lines)


def format_dpo_pair(prompt: str, chosen: str, rejected: str) -> str:
    """Format a DPO preference pair as JSON.

    Args:
        prompt: The shared input prompt for both responses.
        chosen: The preferred (higher-quality) response.
        rejected: The dispreferred (lower-quality or harmful) response.

    Returns:
        Pretty-printed JSON string.
    """
    example = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }
    return json.dumps(example, indent=2, ensure_ascii=False)


def update_sft_outputs(instruction: str, context: str, output: str) -> tuple[str, str]:
    """Compute both SFT format outputs at once."""

    alpaca = format_sft_alpaca(instruction, context, output)
    chatml = format_sft_chatml(instruction, context, output)

    return alpaca, chatml


# --- Build Gradio UI ---

with gr.Blocks(title='Fine-tuning and alignment demo') as demo:

    gr.Markdown("""
    # Fine-tuning and alignment demo

    **Lesson 50 - Fine-tuning, RLHF, and model alignment**

    This demo has two parts:
    - **Model comparison** - see the real behavioral difference between a base model and an instruction-tuned model
    - **Dataset formatter** - explore what SFT and DPO training data actually looks like
    """)

    with gr.Tabs():

        # ── Tab 1: Model Comparison ─────────────────────────────────────────
        with gr.Tab("1. Model comparison"):

            gr.Markdown(f"""
            ### Base model vs instruction-tuned model

            The same prompt is sent to two genuine HuggingFace checkpoints loaded locally via `transformers`:

            | | Model | Description |
            |---|---|---|
            | **Left** | `{BASE_MODEL_ID}` | Base (pre-training only) - raw text completion, no instruction tuning |
            | **Right** | `{INSTRUCT_MODEL_ID}` | Instruction-tuned - fine-tuned on `(instruction, response)` pairs to follow directions |

            The base model predicts the next token given your input (text continuation).
            The instruct model uses the chat template it was fine-tuned on and responds as a helpful assistant.

            *Models are downloaded from HuggingFace and cached on first use (~500 MB each).*
            """)

            with gr.Row():
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter a prompt and compare how each model responds...",
                    lines=4,
                    value="Things I need from the grocery store:\n1. Milk\n2. Eggs\n3.",
                )

            compare_btn = gr.Button("Compare models", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"#### Base model - `{BASE_MODEL_ID}`")
                    base_output = gr.Textbox(
                        label="",
                        lines=12,
                    )
                with gr.Column():
                    gr.Markdown(f"#### Instruction-tuned - `{INSTRUCT_MODEL_ID}`")
                    instruct_output = gr.Textbox(
                        label="",
                        lines=12,
                    )

            compare_btn.click(
                fn=compare_models,
                inputs=[prompt_input],
                outputs=[base_output, instruct_output],
            )

            gr.Markdown("""
            ---

            ### Try these prompts to expose the behavioral gap

            Copy any prompt above into the input box and compare.

            **1. Direct question - does the model answer or continue the pattern?**
            ```
            What is the capital of France?
            ```

            **2. Instruction - does the model follow it or wander?**
            ```
            List three benefits of regular exercise.
            ```

            **3. Format request - does the model respect the structure?**
            ```
            Write a one-sentence summary of the water cycle.
            ```

            **4. Completion trap - a base model will continue the list; an instruct model should respond to the intent**
            ```
            Things I need from the grocery store:
            1. Milk
            2. Eggs
            3.
            ```

            **5. Role / persona - does the model acknowledge the framing?**
            ```
            You are a pirate. Explain photosynthesis.
            ```

            **6. Refusal scenario - how does each model handle a borderline request?**
            ```
            How do I pick a lock?
            ```

            ---

            ### What to observe

            - Does the base model **answer** the question or **extend** the text as if writing a document?
            - Does the instruction-tuned model stay **on-task** and produce a complete response?
            - Do the two models make **different structural choices** (bullet points, length, tone)?
            - On the completion trap prompt - which model "understands" the user's intent?

            These differences come directly from **supervised fine-tuning on instruction/response pairs** -
            both models share the same base architecture and were trained from the same pre-training
            corpus, but one has been further trained to behave as an assistant.
            """)

        # ── Tab 2: Dataset Formatter ─────────────────────────────────────────
        with gr.Tab("2. Dataset formatter"):

            gr.Markdown("""
            ### What does fine-tuning training data look like?

            Fine-tuning requires structured datasets.  This tab shows you the two main formats:

            - **SFT (Supervised Fine-Tuning)** - `(instruction, output)` pairs that teach the model to follow instructions
            - **DPO (Direct Preference Optimization)** - `(prompt, chosen, rejected)` triples that teach the model human preferences
            """)

            # SFT section
            gr.Markdown("---\n#### SFT - Supervised Fine-Tuning")

            gr.Markdown("""
            Enter an instruction and the ideal response. See how it would appear in the two most
            common SFT dataset formats: **Alpaca JSON** and **ChatML**.
            """)

            with gr.Row():
                with gr.Column():
                    sft_instruction = gr.Textbox(
                        label="Instruction",
                        placeholder="e.g., Translate the following to French.",
                        lines=2,
                        value="Summarize the following in one sentence.",
                    )
                    sft_context = gr.Textbox(
                        label="Context / input (optional)",
                        placeholder="Additional input the model should act on...",
                        lines=3,
                        value="The Eiffel Tower was built between 1887 and 1889 as the entrance arch for the 1889 World's Fair in Paris. It was designed by engineer Gustave Eiffel.",
                    )
                    sft_output = gr.Textbox(
                        label="Ideal output",
                        placeholder="The response you want the model to produce...",
                        lines=3,
                        value="The Eiffel Tower is an iron lattice tower built in Paris in 1889 as the entrance arch for that year's World's Fair.",
                    )
                    sft_btn = gr.Button("Format dataset example", variant="primary")

                with gr.Column():
                    sft_alpaca_output = gr.Code(
                        label="Alpaca JSON format",
                        language="json",
                        lines=14,
                    )
                    sft_chatml_output = gr.Code(
                        label="ChatML format",
                        language="markdown",
                        lines=10,
                    )

            sft_btn.click(
                fn=update_sft_outputs,
                inputs=[sft_instruction, sft_context, sft_output],
                outputs=[sft_alpaca_output, sft_chatml_output],
            )

            gr.Markdown("""
            **Alpaca format** - used by the Stanford Alpaca dataset and many community instruction datasets.
            The `input` field is optional; omit it for pure instruction/response pairs.

            **ChatML format** - used by Mistral, Qwen, and many other models.
            The `<|im_start|>` / `<|im_end|>` tokens are special tokens added to the tokenizer vocabulary
            to delimit conversation turns.
            """)

            # DPO section
            gr.Markdown("---\n#### DPO - Direct Preference Optimization")

            gr.Markdown("""
            DPO trains on **preference pairs** - two responses to the same prompt where one is preferred.
            The model learns to increase the probability of the chosen response and decrease the probability
            of the rejected one.
            """)

            with gr.Row():
                with gr.Column():
                    dpo_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="The shared input prompt.",
                        lines=2,
                        value="Explain black holes to a 10-year-old.",
                    )
                    dpo_chosen = gr.Textbox(
                        label="Chosen response (preferred)",
                        placeholder="The better, preferred response...",
                        lines=4,
                        value="Imagine a giant vacuum cleaner in space that's so powerful not even light can escape it. Stars can collapse and become so dense that gravity pulls everything in - that's a black hole!",
                    )
                    dpo_rejected = gr.Textbox(
                        label="Rejected response (dispreferred)",
                        placeholder="The worse or harmful response...",
                        lines=4,
                        value="A black hole is a singularity in spacetime where the escape velocity exceeds the speed of light, resulting from the gravitational collapse of stellar mass objects.",
                    )
                    dpo_btn = gr.Button("Format DPO pair", variant="primary")

                with gr.Column():
                    dpo_output = gr.Code(
                        label="DPO JSON format",
                        language="json",
                        lines=14,
                    )

            dpo_btn.click(
                fn=format_dpo_pair,
                inputs=[dpo_prompt, dpo_chosen, dpo_rejected],
                outputs=[dpo_output],
            )

            gr.Markdown("""
            **Why is the chosen response better here?**
            Both answers are technically correct - but the chosen response uses an analogy
            appropriate for a 10-year-old, while the rejected response uses jargon that would
            confuse a child.  DPO preference pairs capture this kind of nuanced quality signal
            that a simple correct/incorrect label can't express.

            **Tip:** Build your own DPO dataset by writing two responses to the same prompt -
            one that follows the instruction well and one that doesn't.  A dataset of even
            100–200 high-quality pairs can measurably improve model behavior.
            """)

    gr.Markdown("""
    ---

    ## Key takeaways

    1. **Base models are text completers** - they continue patterns, not follow instructions
    2. **Instruction tuning** (SFT) on `(instruction, response)` pairs produces instruction-following behavior
    3. **LoRA/QLoRA** make fine-tuning accessible - ~0.1% of parameters, fraction of the memory
    4. **RLHF/DPO** go further - they train on human *preference* signals, not just correct responses
    5. **Training data format matters** - Alpaca, ChatML, and DPO pairs each serve different training objectives
    """)


# Launch the Gradio app
if __name__ == '__main__':
    demo.launch()
