import os
import re
import gc
import json
import time
import yaml
import faiss
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import r2_score
from transformers import AutoProcessor, SiglipModel, SiglipProcessor
from vllm import LLM, SamplingParams

def json_converter(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.item() if o.numel() == 1 else o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

# =========================================================
# Constants
# =========================================================
CONFIG = "datasets.yaml"
D = 768
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
TRAINED_MODEL_K_SHOTS = 16

FAISS_NLIST = 100
FAISS_NPROBE = 10
FAISS_M = 8
FAISS_BITS = 8
FAISS_NUM_CANDIDATES = 200

MAX_GEN_TOKENS = 20
TEMPERATURE = 0.0
MAX_MODEL_LEN = 32768

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

random.seed(42)

vision_model_inf = None
processor_inf = None


# =========================================================
# Utilities
# =========================================================
def get_dataset_name(task_type: str) -> str:
    mapping = {
        "age_prediction": "utkface",
        "aesthetic_score": "ava",
        "facial_beauty": "FBP5500_v2",
        "modified_image_quality": "KADID-10k",
        "wild_image_quality": "KonIQ-10k",
        "head_num": "jhu_crowd",
    }
    if task_type not in mapping:
        raise ValueError(f"Unsupported task_type: {task_type}")
    return mapping[task_type]


def get_question(task_type: str) -> str:
    mapping = {
        "age_prediction": "What is the age of the person in the last image? Answer with only one number.",
        "head_num": "How many heads are in the last image? Answer with only one number.",
        "aesthetic_score": "What is the aesthetic score of the last image from 0 to 10? Answer with only one number.",
        "facial_beauty": "What is the facial beauty score of the last image from 0 to 5? Answer with only one number.",
        "modified_image_quality": "Rate the last image quality from 0 to 5. Answer with only one number.",
        "wild_image_quality": "Rate the last image quality from 0 to 5. Answer with only one number.",
    }
    if task_type not in mapping:
        raise ValueError(f"Unsupported task_type: {task_type}")
    return mapping[task_type]


def extract_first_number(text):
    if text is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        return float(match.group())
    return None


def build_faiss_index(embeddings_tensor, nlist=FAISS_NLIST, m=FAISS_M, bits=FAISS_BITS):
    n_local, d_local = embeddings_tensor.shape
    embeddings_np = embeddings_tensor.cpu().numpy().astype("float32")

    quantizer = faiss.IndexFlatIP(d_local)
    index = faiss.IndexIVFPQ(quantizer, d_local, nlist, m, bits)
    index.metric_type = faiss.METRIC_INNER_PRODUCT

    print("Training FAISS index...")
    index.train(embeddings_np)
    print("Adding vectors to FAISS index...")
    index.add(embeddings_np)
    index.nprobe = FAISS_NPROBE
    print(f"FAISS index built. Total vectors: {index.ntotal}, nprobe={index.nprobe}")
    return index


def load_siglip():
    global vision_model_inf, processor_inf
    if vision_model_inf is None or processor_inf is None:
        print("Loading SigLIP...")
        vision_model_inf = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(device).eval()
        processor_inf = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")


def encode_image(image_path: str):
    load_siglip()
    image = Image.open(image_path).convert("RGB")
    inputs = processor_inf(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = vision_model_inf.get_image_features(**inputs)
    return F.normalize(image_features, p=2, dim=1)


def build_vllm_prompt_and_images(task_type, model_name, processor, demo_items, query_item):
    query_path, _ = query_item
    question = get_question(task_type)

    demo_imgs = []
    for demo_path, demo_label in demo_items:
        img = Image.open(demo_path).convert("RGB")
        demo_imgs.append((img, demo_label))

    query_img = Image.open(query_path).convert("RGB")
    all_imgs = [img for img, _ in demo_imgs] + [query_img]

    if "phi" in model_name.lower():
        text_parts = []
        for i, (_, label) in enumerate(demo_imgs, start=1):
            text_parts.append(f"<|image_{i}|>\nImage-{i} label: {label}\n")
        q_idx = len(demo_imgs) + 1
        text_parts.append(f"<|image_{q_idx}|>\n{question}")
        raw_text = "".join(text_parts)

        messages = [{"role": "user", "content": raw_text}]
        if hasattr(processor, "tokenizer"):
            prompt = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    elif "internvl" in model_name.lower():
        raw_text = ""
        for i, (_, label) in enumerate(demo_imgs, start=1):
            raw_text += f"<|image_{i}|>\nImage-{i} label: {label}\n"
        q_idx = len(demo_imgs) + 1
        raw_text += f"Image-{q_idx}: <image>\n"
        raw_text += question

        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        messages = [{"role": "user", "content": raw_text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    else:
        content = []
        for i, (_, label) in enumerate(demo_imgs, start=1):
            content.append({"type": "image"})
            content.append({"type": "text", "text": f"Image-{i} label: {label}\n"})
        content.append({"type": "image"})
        content.append({"type": "text", "text": question})

        messages = [{"role": "user", "content": content}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    return prompt, all_imgs


def run_vllm_inference(task_type, model_name, llm, processor, demo_items, query_item, sampling_params):
    try:
        prompt, all_imgs = build_vllm_prompt_and_images(
            task_type, model_name, processor, demo_items, query_item
        )

        outputs = llm.generate(
            [{
                "prompt": prompt,
                "multi_modal_data": {"image": all_imgs},
            }],
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        text = outputs[0].outputs[0].text.strip()
        pred = extract_first_number(text)
        return pred, text

    except Exception as e:
        return None, f"ERROR: {e}"


# =========================================================
# Dueling Q-Network
# =========================================================
class DuelingQNetwork(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, total_samples, max_shots):
        super().__init__()
        self.D = embedding_dim
        self.N = total_samples
        self.max_shots = max_shots

        self.positional_encoding = nn.Embedding(max_shots, self.D)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.D,
            nhead=num_heads,
            dim_feedforward=self.D * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.register_buffer("image_embeddings", torch.zeros(self.N, self.D, dtype=torch.float32))
        self.ln_final = nn.LayerNorm(self.D)

        self.value_head = nn.Linear(self.D, 1)
        self.advantage_head = nn.Linear(self.D, self.D)

    def load_all_image_embeddings(self, embeddings_tensor):
        if embeddings_tensor.shape[1] != self.D:
            raise ValueError(f"Embedding dim mismatch: expected {self.D}, got {embeddings_tensor.shape[1]}")

        loaded_n = embeddings_tensor.shape[0]
        if self.N != loaded_n:
            self.N = loaded_n
            self.register_buffer("image_embeddings", torch.zeros(self.N, self.D, dtype=torch.float32))

        embeddings_tensor = embeddings_tensor.float()
        embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=1)
        self.image_embeddings.data = embeddings_tensor.to(self.image_embeddings.device)

    def get_transformer_context(self, query_emb, demo_embs):
        bsz, k, _ = demo_embs.shape
        k_effective = min(k, self.max_shots)

        tgt_seq = query_emb.unsqueeze(1)

        padding_needed = self.max_shots - k
        if padding_needed > 0:
            padding = torch.zeros(bsz, padding_needed, self.D, device=demo_embs.device)
            demo_embs_padded = torch.cat([demo_embs, padding], dim=1)
        else:
            demo_embs_padded = demo_embs[:, :self.max_shots, :]

        pos = torch.arange(self.max_shots, device=demo_embs_padded.device).unsqueeze(0).expand(bsz, -1)
        memory_seq = demo_embs_padded + self.positional_encoding(pos)

        tgt_padding_mask = torch.zeros(bsz, 1, dtype=torch.bool, device=tgt_seq.device)
        memory_padding_mask = torch.zeros(bsz, self.max_shots, dtype=torch.bool, device=memory_seq.device)
        memory_padding_mask[:, k_effective:] = True

        transformer_out = self.transformer_decoder(
            tgt=tgt_seq,
            memory=memory_seq,
            tgt_mask=None,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )

        context = transformer_out[:, 0, :]
        context = self.ln_final(context)
        return context

    def forward(self, query_emb, demo_embs):
        context = self.get_transformer_context(query_emb, demo_embs)
        state_value = self.value_head(context)
        advantage_query = self.advantage_head(context)
        return state_value, F.normalize(advantage_query, p=2, dim=1)

    def get_q_values_for_candidates(self, state_value, advantage_query, candidate_indices):
        candidate_embs = self.image_embeddings[candidate_indices]
        advantage_query_norm = F.normalize(advantage_query, p=2, dim=-1)
        a_s_a = torch.bmm(
            advantage_query_norm.unsqueeze(1),
            candidate_embs.transpose(1, 2)
        ).squeeze(1)

        q_values = state_value + (a_s_a - a_s_a.mean(dim=1, keepdim=True))
        return q_values


# =========================================================
# Selection methods
# =========================================================
def select_anchor(query_emb, faiss_index, train_size):
    query_np = query_emb.cpu().numpy().astype("float32")

    d_self, i_self = faiss_index.search(query_np, k=1)
    query_idx_in_train = i_self[0, 0] if d_self[0, 0] > 0.999 else -1

    _, i_top2 = faiss_index.search(query_np, k=2)
    potential_anchor = (
        i_top2[0, 1]
        if i_top2[0, 0] == query_idx_in_train and query_idx_in_train != -1 and i_top2.shape[1] > 1
        else i_top2[0, 0]
    )

    if potential_anchor == query_idx_in_train or potential_anchor < 0 or potential_anchor >= train_size:
        valid = [i for i in range(train_size) if i != query_idx_in_train]
        anchor_idx = random.choice(valid) if valid else 0
    else:
        anchor_idx = potential_anchor

    return anchor_idx, query_idx_in_train


def select_demonstrations_random(anchor_idx, train_size, k_shots):
    candidates = [i for i in range(train_size) if i != anchor_idx]
    k = min(k_shots-1, len(candidates))
    return [anchor_idx] + random.sample(candidates, k)


def select_demonstrations_knn(query_emb, faiss_index, anchor_idx, query_idx_in_train, k_shots):
    query_np = query_emb.cpu().numpy().astype("float32")
    _, i_knn = faiss_index.search(query_np, k=k_shots + 8)

    neighbors = list(i_knn[0])
    filtered = []
    invalid = {anchor_idx}
    if query_idx_in_train != -1:
        invalid.add(query_idx_in_train)

    for idx in neighbors:
        if idx not in invalid and idx >= 0:
            filtered.append(idx)
        if len(filtered) == k_shots-1:
            break

    return [anchor_idx] + filtered


def select_demonstrations_dqn(query_emb, policy_net, faiss_index, image_embeddings, anchor_idx, query_idx_in_train, k_shots):
    selected = [anchor_idx]
    n_train = image_embeddings.shape[0]

    for _ in range(k_shots-1):
        with torch.no_grad():
            demo_tensor = image_embeddings[torch.tensor(selected, device=image_embeddings.device)].unsqueeze(0).to(device)
            v, a_query = policy_net(query_emb, demo_tensor)

            search_query = a_query.detach().cpu().numpy().astype("float32")
            _, i_ann = faiss_index.search(search_query, FAISS_NUM_CANDIDATES)
            candidate_indices = torch.tensor(i_ann, device=device)

            q_values = policy_net.get_q_values_for_candidates(v, a_query, candidate_indices)

            invalid = set(selected)
            if query_idx_in_train != -1:
                invalid.add(query_idx_in_train)

            for j, idx in enumerate(candidate_indices[0]):
                idx_val = idx.item()
                if idx_val in invalid or idx_val < 0 or idx_val >= n_train:
                    q_values[0, j] = -float("inf")

            if torch.all(q_values[0] == -float("inf")):
                break

            best_idx = torch.argmax(q_values, dim=1).item()
            next_action = candidate_indices[0, best_idx].item()
            selected.append(next_action)

    return selected


# =========================================================
# Evaluation
# =========================================================
def compute_metrics(true_labels, pred_labels):
    valid = [(t, p) for t, p in zip(true_labels, pred_labels) if p is not None]
    if len(valid) == 0:
        return None

    y_true = np.array([t for t, p in valid], dtype=float)
    y_pred = np.array([p for t, p in valid], dtype=float)

    return {
        "mean_absolute_error": float(np.mean(np.abs(y_true - y_pred))),
        "root_mean_squared_error": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mean_difference": float(np.mean(y_true - y_pred)),
        "r2_score": float(r2_score(y_true, y_pred)),
        "valid_predictions": int(len(valid)),
    }


def evaluate_one_method(
    method_name,
    task_type,
    llm,
    processor,
    inf_model_name,
    policy_net,
    train_dataset,
    train_embeddings,
    faiss_index,
    test_dataset,
    eval_indices,
    k_shots,
):
    print(f"\n--- Evaluating {method_name.upper()} | k={k_shots} ---")

    all_true = []
    all_pred = []
    retrievals = []
    qualitative = []

    for idx in tqdm(eval_indices, desc=f"{method_name}-k{k_shots}"):
        query_item = test_dataset[idx]
        query_path, query_label = query_item

        try:
            query_emb = encode_image(query_path)
            anchor_idx, query_idx_in_train = select_anchor(query_emb, faiss_index, len(train_dataset))

            if method_name == "random":
                selected_indices = select_demonstrations_random(anchor_idx, len(train_dataset), k_shots)
            elif method_name == "knn":
                selected_indices = select_demonstrations_knn(query_emb, faiss_index, anchor_idx, query_idx_in_train, k_shots)
            elif method_name == "dqn":
                selected_indices = select_demonstrations_dqn(
                    query_emb=query_emb,
                    policy_net=policy_net,
                    faiss_index=faiss_index,
                    image_embeddings=train_embeddings,
                    anchor_idx=anchor_idx,
                    query_idx_in_train=query_idx_in_train,
                    k_shots=k_shots,
                )
            else:
                raise ValueError(f"Unknown method: {method_name}")

            demo_items = [train_dataset[j] for j in selected_indices]
            pred, raw_text = run_vllm_inference(
                task_type=task_type,
                model_name=inf_model_name,
                llm=llm,
                processor=processor,
                demo_items=demo_items,
                query_item=query_item,
                sampling_params=SamplingParams(
                    temperature=TEMPERATURE,
                    max_tokens=MAX_GEN_TOKENS,
                ),
            )

            all_true.append(query_label)
            all_pred.append(pred)

            retrievals.append({
                "query_image": os.path.basename(query_path),
                "true_label": query_label,
                "selected_demo_indices": selected_indices,
                "selected_demo_images": [os.path.basename(train_dataset[j][0]) for j in selected_indices],
                "selected_demo_labels": [train_dataset[j][1] for j in selected_indices],
                "predicted_label": pred,
                "raw_response": raw_text,
            })

            if len(qualitative) < 5:
                qualitative.append({
                    "query_image": os.path.basename(query_path),
                    "true_label": query_label,
                    "selected_demo_labels": [train_dataset[j][1] for j in selected_indices],
                    "predicted_label": pred,
                    "raw_response": raw_text,
                })

        except Exception as e:
            print(f"Skipping sample {idx}: {e}")
            all_true.append(query_label)
            all_pred.append(None)

    metrics = compute_metrics(all_true, all_pred)
    if metrics is not None:
        metrics["method"] = method_name
        metrics["total_samples"] = len(eval_indices)

    return metrics, qualitative, retrievals


def print_metrics_table(results):
    print("\n" + "=" * 92)
    print(f"| {'Method':<10} | {'K':<4} | {'MAE':<10} | {'RMSE':<10} | {'R2':<10} | {'Valid %':<8} |")
    print("-" * 92)

    for k_name, method_dict in results.items():
        k_val = k_name.replace("_shots", "")
        for method_name, payload in method_dict.items():
            m = payload["metrics"]
            if m is None:
                print(f"| {method_name:<10} | {k_val:<4} | {'NA':<10} | {'NA':<10} | {'NA':<10} | {'NA':<12} | {'0.0':<8} |")
            else:
                valid_pct = 100.0 * m["valid_predictions"] / max(1, m["total_samples"])
                print(
                    f"| {method_name:<10} | {k_val:<4} | "
                    f"{m['mean_absolute_error']:<10.3f} | "
                    f"{m['root_mean_squared_error']:<10.3f} | "
                    f"{m['r2_score']:<10.3f} | "
                    f"{valid_pct:<8.1f} |"
                )
    print("=" * 92)
    print()


# =========================================================
# Main
# =========================================================
def main(args):
    cfg = yaml.safe_load(open(CONFIG, "r"))
    dataset_root = cfg["datasets"][args.task_type]["root"]
    dataset_name = get_dataset_name(args.task_type)

    train_dataset_path = os.path.join(dataset_root, "train_dataset.pt")
    train_embedding_path = os.path.join(dataset_root, "train_embeddings.pt")
    test_dataset_path = os.path.join(dataset_root, "test_dataset.pt")

    print("Loading datasets...")
    train_dataset = torch.load(train_dataset_path)
    train_embeddings = torch.load(train_embedding_path).to(device).float()
    train_embeddings = F.normalize(train_embeddings, p=2, dim=1)
    test_dataset = torch.load(test_dataset_path)

    print(f"Train size: {len(train_dataset)}")
    print(f"Test size : {len(test_dataset)}")

    faiss_index = build_faiss_index(train_embeddings)

    print("Loading vLLM model...")
    llm = LLM(
        model=args.inf_model_name,
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": TRAINED_MODEL_K_SHOTS + 1},
    )

    processor = AutoProcessor.from_pretrained(args.inf_model_name, trust_remote_code=True)
    if "phi" in args.inf_model_name.lower():
        processor = AutoProcessor.from_pretrained(
            args.inf_model_name,
            trust_remote_code=True,
            num_crops=1,
        )

    policy_net = None
    if args.eval_dqn:
        print(f"Loading DQN checkpoint from {args.model_checkpoint}")
        policy_net = DuelingQNetwork(
            embedding_dim=D,
            num_heads=TRANSFORMER_HEADS,
            num_layers=TRANSFORMER_LAYERS,
            total_samples=len(train_dataset),
            max_shots=TRAINED_MODEL_K_SHOTS,
        ).to(device)

        checkpoint = torch.load(args.model_checkpoint, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        if "image_embeddings" in state_dict:
            del state_dict["image_embeddings"]

        policy_net.load_state_dict(state_dict, strict=False)
        policy_net.load_all_image_embeddings(train_embeddings)
        policy_net.eval()

    num_samples = min(args.eval_num_samples, len(test_dataset))
    eval_indices = random.sample(range(len(test_dataset)), num_samples)

    eval_results = {}
    retrieval_results = {}

    for k_shots in args.eval_k_shots:
        key = f"{k_shots}_shots"
        eval_results[key] = {}
        retrieval_results[key] = {}

        if args.eval_random:
            metrics, qualitative, retrievals = evaluate_one_method(
                method_name="random",
                task_type=args.task_type,
                llm=llm,
                processor=processor,
                inf_model_name=args.inf_model_name,
                policy_net=None,
                train_dataset=train_dataset,
                train_embeddings=train_embeddings,
                faiss_index=faiss_index,
                test_dataset=test_dataset,
                eval_indices=eval_indices,
                k_shots=k_shots,
            )
            eval_results[key]["random"] = {
                "metrics": metrics,
                "qualitative": qualitative,
            }
            retrieval_results[key]["random"] = retrievals

        if args.eval_knn:
            metrics, qualitative, retrievals = evaluate_one_method(
                method_name="knn",
                task_type=args.task_type,
                llm=llm,
                processor=processor,
                inf_model_name=args.inf_model_name,
                policy_net=None,
                train_dataset=train_dataset,
                train_embeddings=train_embeddings,
                faiss_index=faiss_index,
                test_dataset=test_dataset,
                eval_indices=eval_indices,
                k_shots=k_shots,
            )
            eval_results[key]["knn"] = {
                "metrics": metrics,
                "qualitative": qualitative,
            }
            retrieval_results[key]["knn"] = retrievals

        if args.eval_dqn:
            metrics, qualitative, retrievals = evaluate_one_method(
                method_name="dqn",
                task_type=args.task_type,
                llm=llm,
                processor=processor,
                inf_model_name=args.inf_model_name,
                policy_net=policy_net,
                train_dataset=train_dataset,
                train_embeddings=train_embeddings,
                faiss_index=faiss_index,
                test_dataset=test_dataset,
                eval_indices=eval_indices,
                k_shots=k_shots,
            )
            eval_results[key]["dqn"] = {
                "metrics": metrics,
                "qualitative": qualitative,
            }
            retrieval_results[key]["dqn"] = retrievals

    print_metrics_table(eval_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.inf_model_name.split("/")[-1]
    save_dir = os.path.join("eval_res", args.task_type, model_tag, f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    eval_path = os.path.join(save_dir, f"{dataset_name}_eval_results.json")
    retrieval_path = os.path.join(save_dir, f"{dataset_name}_retrieval_results.json")

    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=4, default=json_converter)

    with open(retrieval_path, "w") as f:
        json.dump(retrieval_results, f, indent=4, default=json_converter)

    print(f"Saved eval results to      : {eval_path}")
    print(f"Saved retrieval results to : {retrieval_path}")

    print("Cleanup...")
    for name in ["policy_net", "llm", "processor", "faiss_index", "train_embeddings"]:
        obj = locals().get(name)
        if obj is not None:
            del obj

    global vision_model_inf, processor_inf
    if vision_model_inf is not None:
        del vision_model_inf
        vision_model_inf = None
    if processor_inf is not None:
        del processor_inf
        processor_inf = None

    torch.cuda.empty_cache()
    gc.collect()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN / kNN / Random demonstration selection")

    parser.add_argument("--task_type", type=str, required=True)
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to trained DQN checkpoint (.pth)")
    parser.add_argument("--inf_model_name", type=str, required=True, help="vLLM model used for inference")

    parser.add_argument("--eval_num_samples", type=int, default=200)
    parser.add_argument("--eval_k_shots", type=int, nargs="+", default=[1, 2, 4, 8, 16])

    parser.add_argument("--eval_random", action="store_true")
    parser.add_argument("--eval_knn", action="store_true")
    parser.add_argument("--eval_dqn", action="store_true")

    args = parser.parse_args()
    main(args)