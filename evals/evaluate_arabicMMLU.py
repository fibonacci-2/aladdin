import os
import torch
import torch.nn.functional as F
import tiktoken
from transformers import GPT2LMHeadModel
import pandas as pd

# Assume DATA_CACHE_DIR is defined elsewhere
DATA_CACHE_DIR = "./data/evals/"

enc = tiktoken.get_encoding("gpt2")

def render_example_arabic_mmlu(example):
    """
    Given the ArabicMMLU example as a dictionary, render it as torch tensors:
    - tokens (the tokens of context + completion, of size 5xN, as there are up to 5 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    question = example["Question"]
    context = example.get("Context", "")  # Context may be empty
    answer_key = example["Answer Key"]
    
    # Convert answer key (A-E) to numeric label (0-4)
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    label = label_map[answer_key]
    
    # Gather all options, filtering out empty ones
    options = []
    for i in range(1, 6):
        option_key = f"Option {i}"
        if option_key in example and example[option_key] and str(example[option_key]).strip():
            options.append(str(example[option_key]))
    
    # Construct the prompt: question + context (if available)
    if context:
        prompt = f"{question} {context}"
    else:
        prompt = question
    
    # gather up all the tokens
    ctx_tokens = enc.encode(prompt)
    
    tok_rows = []
    mask_rows = []
    
    for option in options:
        # Note: prepending " " because GPT-2 tokenizer, same as HellaSwag
        option_tokens = enc.encode(" " + option)
        tok_rows.append(ctx_tokens + option_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(option_tokens))
    
    # Handle variable number of options (some questions may have fewer than 5 options)
    num_options = len(options)
    if num_options == 0:
        # Return empty tensors if no valid options
        return {}, torch.zeros((0, 0), dtype=torch.long), torch.zeros((0, 0), dtype=torch.long), label
    
    max_len = max(len(row) for row in tok_rows)
    
    tokens = torch.zeros((num_options, max_len), dtype=torch.long)
    mask = torch.zeros((num_options, max_len), dtype=torch.long)
    
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)
    
    # data dict for debugging/reproducibility
    data = {
        "label": label,
        "ctx_tokens": ctx_tokens,
        "option_tokens": [enc.encode(" " + opt) for opt in options],
        "question": question,
        "context": context,
        "options": options
    }
    
    return data, tokens, mask, label

def iterate_examples_arabic_mmlu(split):
    """
    Iterate through ArabicMMLU examples for the given split
    """
    # Assuming the data is available as CSV files
    csv_path = os.path.join(DATA_CACHE_DIR, f"ArabicMMLU_{split}.csv")
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        example = row.to_dict()
        yield example

def get_most_likely_row(tokens, mask, logits):
    """
    Core evaluation function used by both training and standalone eval
    """
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

@torch.no_grad()
def evaluate_arabic_mmlu(model_type, device, split="val"):
    """
    Standalone evaluation function for ArabicMMLU dataset
    """
    torch.set_float32_matmul_precision('high')  # use tf32
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    # model = torch.compile(model) # optionally torch compile the model

    num_correct_norm = 0
    num_total = 0
    
    for example in iterate_examples_arabic_mmlu(split):
        _, tokens, mask, label = render_example_arabic_mmlu(example)
        
        # Skip examples with no valid options
        if tokens.size(0) == 0:
            continue
            
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        logits = model(tokens).logits
        
        # use the shared evaluation function
        pred_norm = get_most_likely_row(tokens, mask, logits)

        # accumulate stats
        num_total += 1
        num_correct_norm += int(pred_norm == label)
        
        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # debug: pretty print a few examples, and the losses in each case
        if num_total < 10:
            print("---")
            print(f"Question: {example['Question']}")
            if example.get('Context'):
                print(f"Context: {example['Context']}")
            print(f"Options:")
            options = []
            for i in range(1, 6):
                option_key = f"Option {i}"
                if option_key in example and example[option_key] and str(example[option_key]).strip():
                    options.append(str(example[option_key]))
            
            # Calculate losses for display
            shift_logits = (logits[..., :-1, :]).contiguous()
            shift_tokens = (tokens[..., 1:]).contiguous()
            shift_losses = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                         shift_tokens.view(-1), reduction='none')
            shift_losses = shift_losses.view(tokens.size(0), -1)
            shift_mask = (mask[..., 1:]).contiguous()
            masked_shift_losses = shift_losses * shift_mask
            sum_loss = masked_shift_losses.sum(dim=1)
            avg_loss = sum_loss / shift_mask.sum(dim=1)
            
            for i, option in enumerate(options):
                loss_val = avg_loss[i].item() if i < len(options) else float('inf')
                print(f"{i} (loss: {loss_val:.4f}) {option}")
            print(f"predicted: {pred_norm}, actual: {label}")

    print(f"Final ArabicMMLU Results:")
    print(f"Normalized accuracy: {num_correct_norm}/{num_total} = {num_correct_norm/num_total:.4f}")
    return num_correct_norm / num_total if num_total > 0 else 0.0

# Alternative function that takes a model instance instead of model_type
@torch.no_grad()
def evaluate_arabic_mmlu_model(model, device, split="val"):
    """
    Evaluate ArabicMMLU using an existing model instance
    """
    num_correct_norm = 0
    num_total = 0
    
    for example in iterate_examples_arabic_mmlu(split):
        _, tokens, mask, label = render_example_arabic_mmlu(example)
        
        if tokens.size(0) == 0:
            continue
            
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        logits = model(tokens).logits
        pred_norm = get_most_likely_row(tokens, mask, logits)

        num_total += 1
        num_correct_norm += int(pred_norm == label)
        
        if num_total % 100 == 0:
            print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

    acc_norm = num_correct_norm / num_total if num_total > 0 else 0.0
    print(f"ArabicMMLU accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
    return acc_norm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    parser.add_argument("-s", "--split", type=str, default="val", help="the split to evaluate on")
    args = parser.parse_args()
    evaluate_arabic_mmlu(args.model_type, args.device, args.split)
