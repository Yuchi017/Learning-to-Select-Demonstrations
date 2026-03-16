# %% Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoProcessor, SiglipModel, SiglipProcessor
from PIL import Image
import numpy as np
import faiss
import os
import random
import re  # For parsing numeric outputs from VLM responses
import time
from tqdm import tqdm  # Progress bar
import gc
import math
from collections import deque, namedtuple
import argparse
import yaml
from vllm import LLM, SamplingParams

# %% Hyperparameters & Constants
# Dataset configuration
CONFIG = "datasets.yaml"

# N is determined by dataset size at runtime
D = 768  # SigLIP base image embedding dimension

# --- RL / DQN hyperparameters ---
K_SHOTS = 16
GAMMA = 0.99                # Discount factor
REPLAY_BUFFER_SIZE = 50000  # Maximum replay buffer size
LEARN_START_STEPS = 1000    # Start optimization after collecting this many transitions
BATCH_SIZE = 32             # DQN optimization batch size
LEARNING_RATE = 5e-6        # Lower LR for more stable DQN training
EPS_START = 0.9             # Initial epsilon for epsilon-greedy exploration
EPS_END = 0.05              # Final epsilon
# EPS_END = 0.01            # Alternative smaller final epsilon
EPS_DECAY_STEPS = 100000    # Number of steps for epsilon decay
TAU = 0.005                 # Soft update rate for target network
LOG_FREQ = 2                # Log every N steps
CHECKPOINT_FREQ = 200       # Save checkpoint every N steps
REWARD_SCALE = 10.0         # Scale rewards down for training stability

# --- Transformer-based Q-network architecture ---
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2

# --- FAISS retrieval settings ---
FAISS_NLIST = 100
FAISS_NPROBE = 10
FAISS_M = 8
FAISS_BITS = 8
FAISS_NUM_CANDIDATES = 200  # Number of ANN candidates considered during action selection

# --- VLM generation settings ---
MAX_GEN_TOKENS = 20
TEMPERATURE = 0.7
MAX_MODEL_LEN = 32768

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')


def build_faiss_index(embeddings_tensor, nlist=FAISS_NLIST, m=FAISS_M, bits=FAISS_BITS):
    N_local, D_local = embeddings_tensor.shape
    embeddings_np = embeddings_tensor.cpu().numpy().astype('float32')
    quantizer = faiss.IndexFlatIP(D_local)
    index = faiss.IndexIVFPQ(quantizer, D_local, nlist, m, bits)
    index.metric_type = faiss.METRIC_INNER_PRODUCT
    print(f"Training FAISS index...")
    if embeddings_np.shape[0] == 0:
        raise ValueError("Cannot train FAISS index with 0 vectors.")
    index.train(embeddings_np)
    print("Adding vectors to FAISS index...")
    index.add(embeddings_np)
    index.nprobe = FAISS_NPROBE
    print(f"FAISS index built. Total vectors: {index.ntotal}, nprobe={index.nprobe}")
    return index


# %% Dueling Q-Network
# Uses a query-centric Transformer decoder:
# - query embedding is used as tgt
# - selected demo embeddings are used as memory
class DuelingQNetwork(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, total_samples, max_shots):
        super().__init__()
        self.D = embedding_dim
        self.N = total_samples
        self.max_shots = max_shots

        # Positional encoding for the demo sequence (used as decoder memory)
        self.positional_encoding = nn.Embedding(max_shots, self.D)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.D, nhead=num_heads, dim_feedforward=self.D * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.register_buffer("image_embeddings", torch.zeros(self.N, self.D, dtype=torch.float32))
        self.ln_final = nn.LayerNorm(self.D)

        # Dueling heads
        self.value_head = nn.Linear(self.D, 1)   # V(s)
        self.advantage_head = nn.Linear(self.D, self.D)  # Advantage query vector A(s, *)

    def load_all_image_embeddings(self, embeddings_tensor):
        if embeddings_tensor.shape[1] != self.D:
            raise ValueError(f"Embedding dim mismatch! Expected {self.D}, got {embeddings_tensor.shape[1]}")

        loaded_N = embeddings_tensor.shape[0]
        if self.N != loaded_N:
            print(f"Updating N from {self.N} to {loaded_N}")
            self.N = loaded_N
            self.register_buffer("image_embeddings", torch.zeros(self.N, self.D, dtype=torch.float32))

        if not isinstance(embeddings_tensor, torch.Tensor):
            embeddings_tensor = torch.tensor(embeddings_tensor, dtype=torch.float32)
        elif embeddings_tensor.dtype != torch.float32:
            embeddings_tensor = embeddings_tensor.float()

        embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=1)
        self.image_embeddings.data = embeddings_tensor.to(self.image_embeddings.device)
        print(f"Loaded image embeddings into model buffer. Shape: {self.image_embeddings.shape}")

    def get_transformer_context(self, query_emb, demo_embs):
        """
        Query-centric decoder:
        - query_emb: [B, D], used as tgt
        - demo_embs: [B, K, D], used as memory
        """
        B, K, _ = demo_embs.shape
        K_effective = min(K, self.max_shots)

        # 1. Prepare target sequence from the query
        tgt_seq = query_emb.unsqueeze(1)  # [B, 1, D]

        # 2. Pad or truncate demo sequence to max_shots
        padding_needed = self.max_shots - K
        if padding_needed > 0:
            padding = torch.zeros(B, padding_needed, self.D, device=demo_embs.device)
            demo_embs_padded = torch.cat([demo_embs, padding], dim=1)  # [B, max_shots, D]
        else:
            demo_embs_padded = demo_embs[:, :self.max_shots, :]

        # 3. Add positional encoding to the memory sequence
        pos = torch.arange(self.max_shots, device=demo_embs_padded.device).unsqueeze(0).expand(B, -1)
        memory_seq = demo_embs_padded + self.positional_encoding(pos)

        # 4. Query-side masks
        tgt_causal_mask = None
        tgt_padding_mask = torch.zeros(B, 1, dtype=torch.bool, device=tgt_seq.device)

        # 5. Memory-side padding mask
        memory_padding_mask = torch.zeros(B, self.max_shots, dtype=torch.bool, device=memory_seq.device)
        memory_padding_mask[:, K_effective:] = True

        # 6. Transformer decoding
        transformer_out = self.transformer_decoder(
            tgt=tgt_seq,
            memory=memory_seq,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask
        )

        # 7. Final context vector from the single output token
        context_vector = transformer_out[:, 0, :]  # [B, D]
        context_vector = self.ln_final(context_vector)
        return context_vector

    def forward(self, query_emb, demo_embs):
        context = self.get_transformer_context(query_emb, demo_embs)
        state_value = self.value_head(context)  # [B, 1]
        advantage_query = self.advantage_head(context)  # [B, D]
        return state_value, F.normalize(advantage_query, p=2, dim=1)

    def get_q_values_for_candidates(self, state_value, advantage_query, candidate_indices):
        B, num_candidates = candidate_indices.shape
        candidate_embs = self.image_embeddings[candidate_indices]

        advantage_query_norm = F.normalize(advantage_query, p=2, dim=-1)
        A_s_a = torch.bmm(advantage_query_norm.unsqueeze(1), candidate_embs.transpose(1, 2)).squeeze(1)

        q_values = state_value + (A_s_a - A_s_a.mean(dim=1, keepdim=True))
        return q_values


# %% Replay Buffer
Transition = namedtuple(
    'Transition',
    ('query_idx', 'demo_indices', 'action', 'reward', 'next_query_idx', 'next_demo_indices', 'done')
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Store one transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# %% VLM Environment
# Reward is computed from VLM prediction quality and returned incrementally
class VLMEnvironment:
    def __init__(self, task_type, dataset, image_embeddings, faiss_index, model_name, vllm_model, processor, sampling_params, k_shots):
        self.task_type = task_type
        self.dataset = dataset
        self.image_embeddings = image_embeddings
        self.index = faiss_index
        self.model_name = model_name
        self.vllm_model = vllm_model
        self.processor = processor
        self.sampling_params = sampling_params

        self.N = len(dataset)
        self.D = image_embeddings.shape[1]
        self.K_SHOTS = k_shots

        self.query_idx = -1
        self.true_query_age = -1
        self.selected_indices = []
        self.current_step_in_ep = 0
        self.last_reward_score = 0.0

    def reset(self):
        self.query_idx = random.randint(0, self.N - 1)
        self.true_query_age = self.dataset[self.query_idx][1]
        query_emb_np = self.image_embeddings[self.query_idx].unsqueeze(0).cpu().numpy().astype('float32')

        # Initialize with a nearest-neighbor anchor demo
        _, I = self.index.search(query_emb_np, k=2)
        anchor_idx = I[0, 1] if I[0, 0] == self.query_idx else I[0, 0]
        if anchor_idx == self.query_idx or anchor_idx < 0 or anchor_idx >= self.N:
            valid_indices = [i for i in range(self.N) if i != self.query_idx]
            anchor_idx = random.choice(valid_indices) if valid_indices else 0

        self.selected_indices = [anchor_idx]
        self.current_step_in_ep = 0

        try:
            self.last_reward_score = self._get_vlm_reward()
        except Exception as e:
            print(f"Error getting baseline reward: {e}")
            self.last_reward_score = -50.0

        # Return state as index-based representation
        return self.query_idx, self.selected_indices

    def _extract_first_number(self, text):
        if text is None:
            return None

        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match:
            return float(match.group())

        return None

    def _build_question(self):
        if self.task_type == "age_prediction":
            return "What is the age of the person in the last image? Answer with only one number."
        elif self.task_type == "aesthetic_score":
            return "What is the aesthetic score of the last image from 0 to 10? Answer with only one number."
        elif self.task_type == "facial_beauty":
            return "What is the facial beauty score of the last image from 0 to 5? Answer with only one number."
        elif self.task_type == "modified_image_quality":
            return "Rate the last image quality from 0 to 5. Answer with only one number."
        elif self.task_type == "wild_image_quality":
            return "Rate the last image quality from 0 to 5. Answer with only one number."
        else:
            return "Answer with only one number."

    def _get_vlm_reward(self):
        query_img_path, query_label = self.dataset[self.query_idx]
        valid_demo_indices = [idx for idx in self.selected_indices if 0 <= idx < self.N]

        if not valid_demo_indices:
            return -50.0

        demo_items = [self.dataset[i] for i in valid_demo_indices]

        try:
            demo_imgs = []
            for demo_path, demo_label in demo_items:
                try:
                    img = Image.open(demo_path).convert("RGB")
                    demo_imgs.append((img, demo_label))
                except Exception as e:
                    print(f"Skipping demo image: {demo_path}, error: {e}")

            if len(demo_imgs) == 0:
                return -50.0

            try:
                query_img = Image.open(query_img_path).convert("RGB")
            except Exception as e:
                print(f"Error: cannot load query image: {e}")
                return -50.0

            all_imgs = [img for img, _ in demo_imgs] + [query_img]
            question = self._build_question()

            # ---------------------------------------------------
            # Phi-family models expect prompt text with <|image_i|> tags
            # ---------------------------------------------------
            if "phi" in self.model_name.lower():
                text_parts = []
                for i, (_, label) in enumerate(demo_imgs, start=1):
                    text_parts.append(f"<|image_{i}|>\nImage-{i} label: {label}\n")

                query_img_idx = len(demo_imgs) + 1
                text_parts.append(f"<|image_{query_img_idx}|>\n{question}")
                query_text = "".join(text_parts)

                messages = [{"role": "user", "content": query_text}]

                if hasattr(self.processor, "tokenizer"):
                    prompt = self.processor.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    prompt = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

            # ---------------------------------------------------
            # InternVL prompt format
            # ---------------------------------------------------
            elif "internvl" in self.model_name.lower():
                raw_question = ""
                for i, (_, label) in enumerate(demo_imgs, start=1):
                    raw_question += f"<|image_{i}|>\nImage-{i} label: {label}\n"
                q_idx = len(demo_imgs) + 1
                raw_question += f"Image-{q_idx}: <image>\n"
                raw_question += question

                tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
                messages = [{"role": "user", "content": raw_question}]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            # ---------------------------------------------------
            # Generic multimodal chat-template path
            # ---------------------------------------------------
            else:
                content = []
                for i, (_, label) in enumerate(demo_imgs, start=1):
                    content.append({"type": "image"})
                    content.append({
                        "type": "text",
                        "text": f"Image-{i} label: {label}\n"
                    })
                content.append({"type": "image"})
                content.append({
                    "type": "text",
                    "text": question
                })
                messages = [{"role": "user", "content": content}]
                prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            # ---------------------------------------------------
            # vLLM inference
            # ---------------------------------------------------
            outputs = self.vllm_model.generate(
                [{
                    "prompt": prompt,
                    "multi_modal_data": {"image": all_imgs},
                }],
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            full_response_text = outputs[0].outputs[0].text.strip()
            pred = self._extract_first_number(full_response_text)

            print(
                f"VLM Raw Response: {pred}",
                f"Full Response: {full_response_text}",
                f"True Label: {query_label}"
            )

            if pred is None:
                print(f"Could not parse numeric answer: {full_response_text}")
                return -50.0

            mae = abs(float(pred) - float(query_label))
            reward = -mae
            return reward

        except Exception as e:
            print(f"Error during VLM reward computation: {e}")
            return -50.0

    def step(self, action_idx):
        invalid_action = False
        if action_idx == self.query_idx or action_idx in self.selected_indices or action_idx < 0 or action_idx >= self.N:
            invalid_action = True

        if invalid_action:
            reward = -50.0  # Penalty for invalid action
            done = True
            return (self.query_idx, self.selected_indices), reward / REWARD_SCALE, done

        self.selected_indices.append(action_idx)
        self.current_step_in_ep += 1
        done = (self.current_step_in_ep == self.K_SHOTS - 1)

        try:
            current_reward_score = self._get_vlm_reward()
        except Exception as e:
            print(f"Error in _get_vlm_reward during step: {e}")
            current_reward_score = -50.0
            done = True

        # Incremental reward = improvement over previous reward
        reward = current_reward_score - self.last_reward_score
        self.last_reward_score = current_reward_score

        return (self.query_idx, self.selected_indices), reward / REWARD_SCALE, done


# %% DQN Helper: epsilon-greedy action selection
def select_action(state_query_idx, state_demo_indices, policy_net, faiss_index, epsilon):
    """
    Select an action with epsilon-greedy policy.
    State is represented by (query_idx, selected_demo_indices).
    """
    if random.random() < epsilon:
        # Exploration: randomly sample from ANN candidates for efficiency
        with torch.no_grad():
            q_emb, d_emb = _get_state_tensors_from_indices([(state_query_idx, state_demo_indices)], policy_net.image_embeddings)
            _, A_query = policy_net(q_emb, d_emb)
            search_query = A_query.detach().cpu().numpy().astype('float32')
            _, I_ann = faiss_index.search(search_query, FAISS_NUM_CANDIDATES)
            candidate_indices = I_ann[0]

        invalid_mask = set([state_query_idx] + state_demo_indices)
        valid_candidates = [idx for idx in candidate_indices if idx not in invalid_mask and 0 <= idx < policy_net.N]

        if not valid_candidates:
            # Fallback: sample from all valid indices
            all_indices = set(range(policy_net.N))
            valid_indices = list(all_indices - invalid_mask)
            if not valid_indices:
                return 0
            return random.choice(valid_indices)

        return random.choice(valid_candidates)

    else:
        # Exploitation: choose candidate with highest Q-value
        with torch.no_grad():
            q_emb, d_emb = _get_state_tensors_from_indices([(state_query_idx, state_demo_indices)], policy_net.image_embeddings)

            # 1. Get state value and advantage query
            V, A_query = policy_net(q_emb, d_emb)

            # 2. Retrieve ANN candidates
            search_query = A_query.detach().cpu().numpy().astype('float32')
            _, I_ann = faiss_index.search(search_query, FAISS_NUM_CANDIDATES)
            candidate_indices_batch = torch.tensor(I_ann, device=device)

            # 3. Compute Q-values for candidates
            q_values = policy_net.get_q_values_for_candidates(V, A_query, candidate_indices_batch)

            # 4. Mask invalid actions
            invalid_mask = set([state_query_idx] + state_demo_indices)
            for i, idx in enumerate(candidate_indices_batch[0]):
                if idx.item() in invalid_mask or idx.item() < 0 or idx.item() >= policy_net.N:
                    q_values[0, i] = -float('inf')

            if torch.all(q_values[0] == -float('inf')):
                # Fallback if all ANN candidates are invalid
                return select_action(state_query_idx, state_demo_indices, policy_net, faiss_index, epsilon=1.0)

            # 5. Return best valid action
            best_candidate_idx = torch.argmax(q_values, dim=1).item()
            return candidate_indices_batch[0, best_candidate_idx].item()


# %% DQN Helper: convert index-based states into embedding tensors
def _get_state_tensors_from_indices(batch_of_indices, all_embs):
    """
    Convert a batch of (query_idx, demo_indices_list) into:
    - query_embs: [B, D]
    - demo_embs:  [B, K, D]
    """
    B = len(batch_of_indices)
    D = all_embs.shape[1]

    query_indices = [item[0] for item in batch_of_indices]
    query_embs = all_embs[query_indices].to(device)

    max_k_in_batch = 0
    for _, demo_list in batch_of_indices:
        max_k_in_batch = max(max_k_in_batch, len(demo_list))

    demo_embs_batch = torch.zeros(B, max_k_in_batch, D, device=device, dtype=torch.float32)

    for i, (_, demo_list) in enumerate(batch_of_indices):
        k = len(demo_list)
        if k > 0:
            valid_indices = [idx for idx in demo_list if 0 <= idx < all_embs.shape[0]]
            if valid_indices:
                indices_tensor = torch.tensor(valid_indices, dtype=torch.long, device=all_embs.device)
                demo_embs_batch[i, :len(valid_indices), :] = all_embs[indices_tensor]

    return query_embs, demo_embs_batch


# %% DQN Helper: one optimization step
def optimize_model(policy_net, target_net, optimizer, memory, faiss_index):
    if len(memory) < LEARN_START_STEPS:
        return None  # Not enough transitions yet

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # 1. Prepare batch tensors
    state_batch_indices = list(zip(batch.query_idx, batch.demo_indices))
    next_state_batch_indices = list(zip(batch.next_query_idx, batch.next_demo_indices))

    action_batch = torch.tensor(batch.action, device=device, dtype=torch.long).view(-1, 1)
    reward_batch = torch.tensor(batch.reward, device=device, dtype=torch.float32).view(-1, 1)
    done_batch = torch.tensor(batch.done, device=device, dtype=torch.float32).view(-1, 1)

    q_emb, d_emb = _get_state_tensors_from_indices(state_batch_indices, policy_net.image_embeddings)
    next_q_emb, next_d_emb = _get_state_tensors_from_indices(next_state_batch_indices, policy_net.image_embeddings)

    # 2. Compute target Q-values using the target network
    with torch.no_grad():
        V_next, A_query_next = target_net(next_q_emb, next_d_emb)

        search_query = A_query_next.detach().cpu().numpy().astype('float32')
        _, I_ann_next = faiss_index.search(search_query, FAISS_NUM_CANDIDATES)
        next_candidate_indices = torch.tensor(I_ann_next, device=device)

        q_values_next = target_net.get_q_values_for_candidates(V_next, A_query_next, next_candidate_indices)

        # Mask invalid next actions
        for b_idx in range(BATCH_SIZE):
            next_query_idx = batch.next_query_idx[b_idx]
            next_demo_indices = batch.next_demo_indices[b_idx]
            valid_next_demos = list(next_demo_indices)[:policy_net.max_shots]
            invalid_mask = set([next_query_idx] + valid_next_demos)
            for k_idx, cand_idx in enumerate(next_candidate_indices[b_idx]):
                if cand_idx.item() in invalid_mask:
                    q_values_next[b_idx, k_idx] = -float('inf')

        max_next_q = q_values_next.max(dim=1, keepdim=True)[0]
        target_q_value = reward_batch + (GAMMA * max_next_q * (1.0 - done_batch))

    # 3. Compute current Q(s, a)
    V_curr, A_query_curr = policy_net(q_emb, d_emb)

    search_query = A_query_curr.detach().cpu().numpy().astype('float32')
    _, I_ann_curr = faiss_index.search(search_query, FAISS_NUM_CANDIDATES)

    # Make sure the actually taken action is included in candidate set
    candidate_indices = torch.tensor(I_ann_curr, device=device)
    action_batch_flat = action_batch.view(-1)
    is_present = (candidate_indices == action_batch_flat[:, None]).any(dim=1)

    for b_idx in range(BATCH_SIZE):
        if not is_present[b_idx]:
            candidate_indices[b_idx, 0] = action_batch_flat[b_idx]

    q_values_curr_all = policy_net.get_q_values_for_candidates(V_curr, A_query_curr, candidate_indices)

    action_mask = (candidate_indices == action_batch)
    action_indices = torch.where(action_mask, 1.0, 0.0).argmax(dim=1, keepdim=True)
    current_q_value = torch.gather(q_values_curr_all, 1, action_indices)

    # 4. Loss
    loss = F.smooth_l1_loss(current_q_value, target_q_value)

    # 5. Backpropagation
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

    # 6. Soft update target network
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)

    return loss.item()


# %% Main DQN training script
def main(args):
    global N, D

    print("--- Phase 1: Setup, Pre-computation, and Data ---")

    cfg = yaml.safe_load(open(CONFIG, "r"))
    data_config = cfg['datasets']
    dataset_path = data_config[args.task_type]['root']

    EMBEDDING_FILE = os.path.join(dataset_path, f"train_embeddings.pt")
    DATASET_FILE = os.path.join(dataset_path, f"train_dataset.pt")

    try:
        if os.path.exists(DATASET_FILE) and os.path.exists(EMBEDDING_FILE):
            print("Found existing data files. Loading...")
            dataset = torch.load(DATASET_FILE)
            image_embeddings = torch.load(EMBEDDING_FILE).to(device).float()
            N = len(dataset)
            if image_embeddings.shape[1] != D:
                raise ValueError("Embedding dim mismatch.")
            print(f"Loaded {N} TRAINING samples. Dim={D}")
        else:
            print(f"ERROR: Training data not found at {DATASET_FILE} or {EMBEDDING_FILE}.")
            print("Please run `prepare_data.py` first.")
            return
    except Exception as e:
        print(f"Fatal Error loading/processing dataset: {e}")
        return

    faiss_index = build_faiss_index(image_embeddings)

    try:
        llm = LLM(
            model=args.model_name,
            tensor_parallel_size=1,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": K_SHOTS + 1},
        )

        processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
        if "phi" in args.model_name.lower():
            processor = AutoProcessor.from_pretrained(
                args.model_name,
                trust_remote_code=True,
                num_crops=1
            )

        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=MAX_GEN_TOKENS,
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return

    print("\n--- Phase 2: Initialize DQN Models, Environment, and Replay Buffer ---")

    policy_net = DuelingQNetwork(
        embedding_dim=D, num_heads=TRANSFORMER_HEADS, num_layers=TRANSFORMER_LAYERS,
        total_samples=N, max_shots=K_SHOTS
    ).to(device)

    target_net = DuelingQNetwork(
        embedding_dim=D, num_heads=TRANSFORMER_HEADS, num_layers=TRANSFORMER_LAYERS,
        total_samples=N, max_shots=K_SHOTS
    ).to(device)

    policy_net.load_all_image_embeddings(image_embeddings)
    target_net.load_all_image_embeddings(image_embeddings)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target net is used only for inference / target computation

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    env = VLMEnvironment(
        task_type=args.task_type,
        dataset=dataset,
        image_embeddings=image_embeddings,
        faiss_index=faiss_index,
        model_name=args.model_name,
        vllm_model=llm,
        processor=processor,
        sampling_params=sampling_params,
        k_shots=K_SHOTS
    )

    memory = ReplayBuffer(REPLAY_BUFFER_SIZE)

    print("\n--- Phase 3: DQN Training Loop ---")
    output_base = f"train_res/{args.task_type}/{args.model_name.split('/')[-1]}/run_{int(time.time())}"
    ckpt_dir = os.path.join(output_base, f"checkpoints")
    log_dir = os.path.join(output_base, f"logs")
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = f"{log_dir}/log.txt"
    print(f"Logging to {log_file_path}")

    (query_idx, demo_indices) = env.reset()
    current_episode_reward = 0.0
    episode_rewards = []
    episode_losses = []

    pbar = tqdm(range(1, args.training_steps + 1), desc="DQN Training")
    start_time = time.time()

    try:
        with open(log_file_path, "w") as log_file:
            for global_step in pbar:
                # 1. Select action
                epsilon = EPS_END + (EPS_START - EPS_END) * \
                          math.exp(-1. * (global_step - LEARN_START_STEPS) / EPS_DECAY_STEPS)
                epsilon = max(EPS_END, epsilon)

                action = select_action(query_idx, demo_indices, policy_net, faiss_index, epsilon)

                # 2. Step environment
                (next_query_idx, next_demo_indices), reward, done = env.step(action)
                current_episode_reward += reward

                # 3. Store transition
                memory.push(query_idx, demo_indices, action, reward, next_query_idx, next_demo_indices, done)

                # 4. Advance state
                query_idx, demo_indices = next_query_idx, next_demo_indices

                # 5. Optimize policy
                loss = optimize_model(policy_net, target_net, optimizer, memory, faiss_index)
                if loss is not None:
                    episode_losses.append(loss)

                # 6. Handle episode termination
                if done:
                    episode_rewards.append(current_episode_reward * REWARD_SCALE)  # Log unscaled reward
                    current_episode_reward = 0.0
                    (query_idx, demo_indices) = env.reset()

                # 7. Periodic logging
                if global_step % LOG_FREQ == 0 and len(episode_rewards) > 0:
                    mean_reward = np.mean(episode_rewards[-100:])
                    mean_loss = np.mean(episode_losses[-100:]) if episode_losses else 0.0
                    elapsed_time = time.time() - start_time
                    fps = int(global_step / elapsed_time) if elapsed_time > 0 else 0

                    log_msg = (
                        f"Step {global_step}/{args.training_steps} | "
                        f"Mean Reward (100ep) = {mean_reward:.3f} | "
                        f"Mean Loss (100ep) = {mean_loss:.4f} | "
                        f"Epsilon = {epsilon:.3f} | FPS: {fps}"
                    )
                    pbar.set_description(log_msg)
                    log_file.write(log_msg + "\n")
                    log_file.flush()

                # 8. Periodic checkpointing
                if global_step % CHECKPOINT_FREQ == 0:
                    chkpt_path = os.path.join(f"{ckpt_dir}", f"step_{global_step}.pth")

                    torch.save({
                        'step': global_step,
                        'model_state_dict': policy_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, chkpt_path)
                    print(f"\nCheckpoint saved to {chkpt_path}")

    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        pbar.close()

        print("Saving final model...")
        model_save_path = os.path.join(ckpt_dir, "final.pth")
        torch.save({
            'step': global_step if 'global_step' in locals() else 0,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_save_path)
        print(f"Final model saved to {model_save_path}")

        print("Cleanup...")
        for name in ["env", "llm", "processor", "faiss_index"]:
            obj = locals().get(name)
            if obj is not None:
                del obj
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "final_model_save_path": model_save_path,
            "embedding_file_path": EMBEDDING_FILE,
            "dataset_file_path": DATASET_FILE,    
        }



# %% Inference function for DQN-based demo selection
def select_demonstrations_age_dqn(query_image_path, policy_net, faiss_index, image_embeddings_tensor, dataset, k_shots):
    policy_net.eval()
    device = next(policy_net.parameters()).device
    N_inf = len(dataset)

    # 1. Compute query embedding using SigLIP
    try:
        global vision_model_inf, processor_inf
        if 'vision_model_inf' not in globals() or 'processor_inf' not in globals():
            print("Loading SigLIP for inference...")
            vision_model_inf = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(device).eval()
            processor_inf = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")

        image = Image.open(query_image_path).convert("RGB")
        inputs = processor_inf(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = vision_model_inf.get_image_features(**inputs)
        query_emb_norm = F.normalize(image_features, p=2, dim=1)
    except Exception as e:
        print(f"Error processing query image {query_image_path}: {e}")
        return [], -1

    # 2. Try to locate query in dataset and choose initial anchor
    query_idx_in_dataset = -1
    anchor_idx = -1
    try:
        query_np = query_emb_norm.cpu().numpy().astype('float32')
        D_self, I_self = faiss_index.search(query_np, k=1)
        if D_self[0, 0] > 0.999:
            query_idx_in_dataset = I_self[0, 0]

        _, I = faiss_index.search(query_np, k=2)
        potential_anchor = I[0, 1] if I[0, 0] == query_idx_in_dataset and I.shape[1] > 1 else I[0, 0]

        if potential_anchor == query_idx_in_dataset or potential_anchor < 0 or potential_anchor >= N_inf:
            valid_anchors = [i for i in range(N_inf) if i != query_idx_in_dataset]
            anchor_idx = random.choice(valid_anchors)
        else:
            anchor_idx = potential_anchor

        selected_indices = [anchor_idx]
    except Exception as e:
        print(f"Error finding anchor: {e}")
        return [], query_idx_in_dataset

    # 3. Greedily select K demonstrations with epsilon = 0
    for k in range(k_shots):
        with torch.no_grad():
            # 1. Build state tensors
            q_idx_tensor = torch.tensor([query_idx_in_dataset if query_idx_in_dataset != -1 else 0], device=device)
            query_emb = image_embeddings_tensor[q_idx_tensor] if query_idx_in_dataset != -1 else query_emb_norm

            indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=image_embeddings_tensor.device)
            demo_embs = image_embeddings_tensor[indices_tensor].unsqueeze(0).to(device)

            # 2. Compute V(s) and A(s, *)
            V, A_query = policy_net(query_emb, demo_embs)

            # 3. Retrieve ANN candidates
            search_query_np = A_query.detach().cpu().numpy().astype('float32')
            _, I_ann = faiss_index.search(search_query_np, FAISS_NUM_CANDIDATES)
            candidate_indices_batch = torch.tensor(I_ann, device=device)

            # 4. Compute Q-values
            q_values = policy_net.get_q_values_for_candidates(V, A_query, candidate_indices_batch)

            # 5. Mask invalid candidates
            invalid_mask = set([query_idx_in_dataset] + selected_indices)
            for i, idx in enumerate(candidate_indices_batch[0]):
                if idx.item() in invalid_mask or idx.item() < 0 or idx.item() >= N_inf:
                    q_values[0, i] = -float('inf')

            if torch.all(q_values[0] == -float('inf')):
                break

            # 6. Choose best candidate
            best_candidate_idx = torch.argmax(q_values, dim=1).item()
            next_action = candidate_indices_batch[0, best_candidate_idx].item()

        selected_indices.append(next_action)

    return selected_indices[1:], query_idx_in_dataset


# %% Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Training for Demonstration Selection")
    parser.add_argument("--task_type", type=str, default="age_prediction", help="Task type (age_prediction, aesthetic_score, etc.)")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-4b-it", help="VLM model name used for reward computation")
    parser.add_argument("--training_steps", type=int, default=20000, help="Total number of DQN training steps")

    args = parser.parse_args()

    # === Training ===
    model_paths = main(args)

    # === Inference example ===
    print("\n--- Inference Example ---")
    final_path_exists = os.path.exists(model_paths["final_model_save_path"])
    embedding_path_exists = os.path.exists(model_paths["embedding_file_path"])
    dataset_path_exists = os.path.exists(model_paths["dataset_file_path"])

    if final_path_exists and embedding_path_exists and dataset_path_exists:
        print("Loading components for inference...")
        try:
            IMAGE_EMBEDDINGS_inf = torch.load(model_paths["embedding_file_path"]).to(device).float()
            dataset_inf = torch.load(model_paths["dataset_file_path"])
            N_inf = len(dataset_inf)
            D_inf = IMAGE_EMBEDDINGS_inf.shape[1]
            if D_inf != D:
                raise ValueError("Dimension mismatch")
            IMAGE_EMBEDDINGS_inf = F.normalize(IMAGE_EMBEDDINGS_inf, p=2, dim=1)
            faiss_index_inf = build_faiss_index(IMAGE_EMBEDDINGS_inf)

            policy_net_inf = DuelingQNetwork(D_inf, TRANSFORMER_HEADS, TRANSFORMER_LAYERS, N_inf, K_SHOTS).to(device)
            checkpoint = torch.load(model_paths["final_model_save_path"], map_location=device)

            model_state_dict = checkpoint['model_state_dict']
            if 'image_embeddings' in model_state_dict:
                del model_state_dict['image_embeddings']

            policy_net_inf.load_state_dict(model_state_dict, strict=False)
            policy_net_inf.load_all_image_embeddings(IMAGE_EMBEDDINGS_inf)
            policy_net_inf.eval()
            print("Trained DQN policy loaded successfully for inference.")

            query_idx_inf_example = 150
            if query_idx_inf_example >= N_inf:
                query_idx_inf_example = 0
            query_image_path_inf, query_label_inf = dataset_inf[query_idx_inf_example]

            print(f"\nSelecting demonstrations for query: {os.path.basename(query_image_path_inf)} (True Age: {query_label_inf})")

            selected_demo_indices, _ = select_demonstrations_age_dqn(
                query_image_path_inf,
                policy_net_inf,
                faiss_index_inf,
                IMAGE_EMBEDDINGS_inf,
                dataset_inf,
                K_SHOTS
            )

            print(f"\nSelected {len(selected_demo_indices)} demonstration indices:")
            print(selected_demo_indices)

            # Optional: call Gemma or another VLM with the selected demos,
            # following the same pattern as in your PPO script.

        except Exception as e:
            print(f"An error occurred during the inference setup: {e}")
            import traceback
            traceback.print_exc()

    else:
        print(f"Training files not found. Run the script first.")