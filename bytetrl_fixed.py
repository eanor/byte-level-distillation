import random
import warnings
import unicodedata
from collections import deque
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from accelerate import PartialState
from accelerate.utils import broadcast_object_list
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
    PreTrainedModel, PreTrainedTokenizerBase,
    Trainer, TrainerCallback, DefaultDataCollator, TrainingArguments
)
from transformers.data.data_collator import DataCollator
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available

import gc
from byte_sampler.src.byte_sampling import ByteConditioning

# ==================== Вспомогательные функции ====================

def pad(tensors, padding_side="right", padding_value=0):
    if not tensors:
        return torch.empty(0)
    max_len = max(t.shape[0] for t in tensors)
    padded = []
    for t in tensors:
        if t.shape[0] < max_len:
            pad_size = (0, max_len - t.shape[0]) if padding_side == "right" else (max_len - t.shape[0], 0)
            padded.append(F.pad(t, pad_size, value=padding_value))
        else:
            padded.append(t)
    return torch.stack(padded)


def split_tensor_dict(tensor_dict, num_splits):
    slices = []
    keys = list(tensor_dict.keys())
    if not keys:
        return [{}] * num_splits
    first_val = tensor_dict[keys[0]]
    batch_size = first_val.shape[0] if isinstance(first_val, torch.Tensor) else (
        len(first_val) if isinstance(first_val, list) else 1
    )
    chunk_size = max(1, batch_size // num_splits)
    for i in range(num_splits):
        slice_dict = {}
        for k in keys:
            val = tensor_dict[k]
            if isinstance(val, (torch.Tensor, list)):
                slice_dict[k] = val[i * chunk_size:(i + 1) * chunk_size]
            else:
                slice_dict[k] = val
        slices.append(slice_dict)
    return slices


class RepeatSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, mini_repeat_count, batch_size, repeat_count, shuffle=True, seed=None):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.shuffle = shuffle
        self.seed = seed if seed is not None else 0
        self.epoch = 0

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(indices), generator=g).tolist()
        repeated = [idx for idx in indices for _ in range(self.mini_repeat_count)]
        batches = [repeated[i:i + self.batch_size] for i in range(0, len(repeated), self.batch_size)]
        final = [idx for batch in batches for _ in range(self.repeat_count) for idx in batch]
        yield from final
        self.epoch += 1

    def __len__(self):
        return len(self.data_source) * self.mini_repeat_count * self.repeat_count


def build_teacher_inputs_from_texts(tokenizer, prompt_texts, completion_texts, device,
                                     max_completion_tokens=256):
    pad_token_id = tokenizer.pad_token_id or 0
    eos_token_id = tokenizer.eos_token_id

    prompt_token_ids = tokenizer(prompt_texts, add_special_tokens=True)["input_ids"]
    completion_token_ids = tokenizer(completion_texts, add_special_tokens=False)["input_ids"]

    sequences, attention_masks, labels_list, prompt_lengths = [], [], [], []
    for p_ids, c_ids in zip(prompt_token_ids, completion_token_ids):
        if eos_token_id is not None and p_ids and p_ids[-1] == eos_token_id:
            p_ids = p_ids[:-1]
        prompt_lengths.append(len(p_ids))

        # обрезаем completion
        c_ids = c_ids[:max_completion_tokens]
        # После строки c_ids = c_ids[:max_completion_tokens]
        # ДОБАВИТЬ: обрезаем до первого EOS включительно

        # ДОБАВИТЬ: обрезаем до первого <|endoftext|> как последовательности токенов
        endoftext_seq = tokenizer.encode("<|endoftext|>", add_special_tokens=False)
        def find_subseq(seq, subseq):
            for i in range(len(seq) - len(subseq) + 1):
                if seq[i:i+len(subseq)] == subseq:
                    return i
            return -1
        pos = find_subseq(c_ids, endoftext_seq)
        if pos != -1:
            c_ids = c_ids[:pos]

        if eos_token_id is not None and eos_token_id in c_ids:
            eos_pos = c_ids.index(eos_token_id)
            c_ids = c_ids[:eos_pos]  # EOS добавится ниже через seq.append(eos_token_id)

        seq = p_ids + c_ids
        if eos_token_id is not None:
            seq.append(eos_token_id)

        seq_t = torch.tensor(seq, dtype=torch.long, device=device)
        sequences.append(seq_t)
        attention_masks.append(torch.ones_like(seq_t))

        labels = seq_t.clone()
        # маскируем только промпт
        labels[:len(p_ids)] = -100
        # НЕ маскируем pad_token_id — если он совпадает с eos, мы потеряем EOS
        # completion уже обрезан, паддинг добавится через padding_value=-100

        labels_list.append(labels)

    input_ids = pad(sequences, padding_side="right", padding_value=pad_token_id)
    attention_mask = pad(attention_masks, padding_side="right", padding_value=0).bool()
    labels = pad(labels_list, padding_side="right", padding_value=-100)
    # паддинг в labels уже -100 из padding_value — дополнительно маскировать не нужно

    teacher_prompt_length = max(prompt_lengths) if prompt_lengths else 0
    return input_ids, labels, attention_mask, teacher_prompt_length


# ==================== XPU-совместимость для ByteConditioning ====================

def _byte_loss_smoke_test(teacher_bc, student_bc):
    """
    - создание BytewiseBatchSampler (инициализирует RadixCacheManager на device модели)
    - add_context с простой строкой
    - get_dists() -> tree_inference -> rcm.query() -> forward модели с KV-кешем
    - scatter_logsumexp с тензорами на device модели
    """
    import traceback
    bc_device = str(teacher_bc.device)
    print(f"[byte_loss] Smoke-test ByteConditioning на device={bc_device}...")
    try:
        sampler = teacher_bc.BytewiseBatchSampler(
            teacher_bc, batch_size=1, filter_tensors=True, do_gc=False
        )
        sampler.add_context([b"test"])
        with torch.no_grad():
            dists = sampler.get_dists()
        assert dists.shape[-1] == 257, f"unexpected dists shape: {dists.shape}"
        del sampler
        _empty_cache()
        print(f"[byte_loss] Smoke-test PASSED на device={bc_device}")
    except Exception as e:
        tb = traceback.format_exc()
        xpu_hints = ["not implemented", "not supported", "XPUError",
                     "RuntimeError", "could not create", "aten::"]
        is_xpu_issue = (bc_device.startswith("xpu")
                        and any(h.lower() in str(e).lower() for h in xpu_hints))
        if is_xpu_issue:
            raise RuntimeError(
                f"\n{'='*70}\n"
                f"ByteConditioning smoke-test FAILED на device={bc_device}\n"
                f"{'='*70}"
            ) from e
        else:
            raise  # не XPU-проблема - пробрасываем как есть

# ==================== ULDLoss ====================

class ULDLoss(nn.Module):
    def __init__(
        self,
        config,
        student_tokenizer=None,
        teacher_tokenizer=None,
        device=None,
        teacher_model=None,   # ИСПРАВЛЕНИЕ #2: принимаем готовую модель, не путь
        student_model=None,   # ИСПРАВЛЕНИЕ #2: принимаем готовую модель, не путь
    ):
        super().__init__()
        self.device = device
        self.crossentropy_weight = config.uld_crossentropy_weight
        self.distillation_weight = config.uld_distillation_weight
        self.student_temperature = config.uld_student_temperature
        self.teacher_temperature = config.uld_teacher_temperature
        self.skip_student_eos = config.uld_skip_student_eos
        self.skip_teacher_eos = config.uld_skip_teacher_eos
        self.use_extended_uld = config.use_extended_uld
        self.ignore_index = -100
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.use_hybrid_loss = getattr(config, "uld_use_hybrid_loss", False)
        self.hybrid_matched_weight = getattr(config, "uld_hybrid_matched_weight", None)
        self.hybrid_unmatched_weight = getattr(config, "uld_hybrid_unmatched_weight", None)
        self.beta = getattr(config, "beta", 1.0)
        self._vocab_mapping = None
        self._teacher_matched_ids = None
        self._student_matched_ids = None
        self.mapping_tensor = None
        self.last_matched_loss = None
        self.last_unmatched_loss = None

        if self.use_hybrid_loss and student_tokenizer and teacher_tokenizer:
            self._initialize_vocabulary_mapping()

        # ИСПРАВЛЕНИЕ #2: ByteConditioning создаётся с уже загруженной моделью
        # Не загружаем веса с диска повторно!
        self.use_byte_loss = getattr(config, "use_byte_loss", False)
        self.teacher_bc = None
        self.student_bc = None

        if self.use_byte_loss and ByteConditioning is not None:
            if teacher_model is not None and teacher_tokenizer is not None:
                self.teacher_bc = ByteConditioning(teacher_model, tokenizer=teacher_tokenizer)
            if student_model is not None and student_tokenizer is not None:
                self.student_bc = ByteConditioning(student_model, tokenizer=student_tokenizer)
            if self.teacher_bc is None or self.student_bc is None:
                warnings.warn(
                    "use_byte_loss=True но модели не переданы. Байтовый лосс отключён."
                )
                self.use_byte_loss = False
            else:
                # ИСПРАВЛЕНИЕ XPU: smoke-test при инициализации - ловим несовместимость
                # с конкретным трейсбэком вместо падения в середине обучения.
                _byte_loss_smoke_test(self.teacher_bc, self.student_bc)

    def _initialize_vocabulary_mapping(self):
        student_vocab = self.student_tokenizer.get_vocab()
        teacher_vocab = self.teacher_tokenizer.get_vocab()
        vocab_mapping = {}
        teacher_matched_ids = set()
        student_matched_ids = set()
        for token_str, teacher_id in teacher_vocab.items():
            if token_str in student_vocab:
                student_id = student_vocab[token_str]
                vocab_mapping[teacher_id] = student_id
                teacher_matched_ids.add(teacher_id)
                student_matched_ids.add(student_id)
        self._vocab_mapping = vocab_mapping
        self._teacher_matched_ids = teacher_matched_ids
        self._student_matched_ids = student_matched_ids
        if self._vocab_mapping:
            teacher_vocab_size = len(self.teacher_tokenizer)
            self.mapping_tensor = torch.full(
                (teacher_vocab_size,), -1, dtype=torch.long,
                device=self.device or "cpu"
            )
            for k, v in self._vocab_mapping.items():
                self.mapping_tensor[k] = v

    def forward(self, student_logits, teacher_logits, student_labels, teacher_labels,
                student_input_ids, teacher_input_ids):
        ce_loss = torch.tensor(0.0, device=student_logits.device)
        if self.crossentropy_weight > 0:
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = student_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            ce_loss = self.crossentropy_weight * loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        dist_loss = self._compute_distillation_loss(
            student_logits, teacher_logits, student_labels, teacher_labels,
            student_input_ids, teacher_input_ids
        )
        return ce_loss + dist_loss

    # -------- Байтовые распределения (ИСПРАВЛЕНИЕ #1, #3, #4, #5) --------

    @staticmethod
    def _compute_byte_dists_from_sampler(
        sampler, suffix_bytes: bytes, device: torch.device
    ) -> torch.Tensor:
        """
        - get_dists() возвращает LOG-PROBS, переводим в вероятности через .exp().
        - Добавляем байты инкрементально, не пересоздавая контекст.
        Возвращает tensor shape (len(suffix_bytes), 257), dtype float32.
        """
        dists_list = []
        for b in suffix_bytes:
            log_probs = sampler.get_dists()  # (1, 257), log-probs на device модели
            # ИСПРАВЛЕНИЕ XPU: НЕ делаем .to(device) на каждом байте -
            # это дорогая синхронизация на XPU. Остаёмся на device сэмплера.
            probs = log_probs[0].exp()  # (257,) вероятности на device модели
            dists_list.append(probs[:256])  # берём только первые 256 байт
            sampler.add_context([bytes([b])])
        if not dists_list:
            return torch.zeros(0, 256, device=device)
        # ИСПРАВЛЕНИЕ XPU: один перенос на target device вместо N в цикле
        result = torch.stack(dists_list)
        if result.device.type != device.type or result.device.index != device.index:
            result = result.to(device)
        return result

    @staticmethod
    def _jsd_from_byte_dists(dist_P: torch.Tensor, dist_Q: torch.Tensor) -> torch.Tensor:
        """JSD между двумя батчами байтовых распределений. shape: (L, 256) -> scalar."""
        eps = 1e-8
        # ИСПРАВЛЕНИЕ XPU: teacher и student могут оказаться на разных device.
        if dist_Q.device != dist_P.device:
            dist_Q = dist_Q.to(dist_P.device)
        P = dist_P.clamp_min(eps)
        Q = dist_Q.clamp_min(eps)
        # нормализуем на случай численных ошибок
        P = P / P.sum(dim=-1, keepdim=True)
        Q = Q / Q.sum(dim=-1, keepdim=True)
        M = 0.5 * (P + Q)
        kl_PM = (P * (P.log() - M.log())).sum(dim=-1)
        kl_QM = (Q * (Q.log() - M.log())).sum(dim=-1)
        return (0.5 * (kl_PM + kl_QM)).mean()

    # -------- Алайнмент (ИСПРАВЛЕНИЕ #7) --------

    def _build_alignment_groups_from_ids(self, student_token_ids, teacher_token_ids):
        """
        используем convert_ids_to_tokens вместо O(N²) decode-loop.
        """
        def normalize_piece(s):
            # убираем replacement character и непечатаемые символы
            # кроме \n и \t которые важны для alignment
            s = s.replace('\ufffd', '')
            return ''.join(c for c in s if unicodedata.category(c) != 'Cc' or c in '\n\t')

        def get_pieces(tokenizer, ids):
            raw = tokenizer.convert_ids_to_tokens(ids)
            pieces = []
            for tok in raw:
                try:
                    decoded = tokenizer.convert_tokens_to_string([tok])
                except Exception:
                    decoded = tok or ""
                decoded = normalize_piece(decoded)
                pieces.append(decoded)
            return pieces

        s_pieces = get_pieces(self.student_tokenizer, student_token_ids)
        t_pieces = get_pieces(self.teacher_tokenizer, teacher_token_ids)

        i = j = 0
        s_buf = t_buf = ""
        s_group, t_group = [], []
        s_groups, t_groups = [], []

        def flush():
            if s_group and t_group:
                s_groups.append(s_group.copy())
                t_groups.append(t_group.copy())

        while i < len(s_pieces) or j < len(t_pieces):
            if s_buf == t_buf and s_buf != "":
                flush()
                s_buf = t_buf = ""
                s_group, t_group = [], []
                continue
            if s_buf == "" and i < len(s_pieces):
                s_buf += s_pieces[i]; s_group.append(i); i += 1; continue
            if t_buf == "" and j < len(t_pieces):
                t_buf += t_pieces[j]; t_group.append(j); j += 1; continue
            if len(s_buf) <= len(t_buf):
                if i < len(s_pieces):
                    s_buf += s_pieces[i]; s_group.append(i); i += 1
                elif j < len(t_pieces):
                    t_buf += t_pieces[j]; t_group.append(j); j += 1
            else:
                if j < len(t_pieces):
                    t_buf += t_pieces[j]; t_group.append(j); j += 1
                elif i < len(s_pieces):
                    s_buf += s_pieces[i]; s_group.append(i); i += 1

        if s_buf == t_buf and (s_group or t_group):
            flush()
        elif s_group or t_group:
            s_groups.append(s_group.copy() if s_group else [])
            t_groups.append(t_group.copy() if t_group else [])

        return s_groups, t_groups

    # -------- Проекция словаря --------

    def _project_teacher_to_student_vocab(self, teacher_prob, target_vocab_size=None):
        device = teacher_prob.device
        if target_vocab_size is None:
            target_vocab_size = len(self.student_tokenizer)
        projected = torch.zeros(target_vocab_size, device=device, dtype=teacher_prob.dtype)
        if self._teacher_matched_ids and self.mapping_tensor is not None:
            mt = self.mapping_tensor.to(device)
            max_teacher_id = min(mt.size(0), teacher_prob.size(-1))
            valid_mask = mt[:max_teacher_id] >= 0
            teacher_ids = torch.arange(max_teacher_id, device=device)[valid_mask]
            student_ids = mt[:max_teacher_id][valid_mask]
            in_range = student_ids < target_vocab_size
            projected[student_ids[in_range]] = teacher_prob[teacher_ids[in_range]]
        return projected

    def _jsd_single_token(self, s_prob, t_prob):
        target_size = s_prob.size(-1)
        t_prob_proj = self._project_teacher_to_student_vocab(t_prob, target_size)
        return GOLDTrainer.generalized_jsd_loss(
            s_prob.unsqueeze(0), t_prob_proj.unsqueeze(0),
            labels=None, beta=self.beta, temperature=1.0,
            reduction="batchmean", logits_are_probs=True
        )

    def _uld_single_token(self, s_prob, t_prob):
        s_sorted = s_prob.sort(descending=True).values
        t_sorted = t_prob.sort(descending=True).values
        sv, tv = s_sorted.size(0), t_sorted.size(0)
        if sv < tv:
            s_sorted = F.pad(s_sorted, (0, tv - sv))
        elif tv < sv:
            t_sorted = F.pad(t_sorted, (0, sv - tv))
        return F.l1_loss(s_sorted, t_sorted, reduction="sum") / s_sorted.size(0)

    # -------- Главный compute_distillation_loss (ИСПРАВЛЕНИЕ #1, #4, #6) --------

    def _compute_distillation_loss(
        self,
        student_logits, teacher_logits,
        student_labels, teacher_labels,
        student_input_ids, teacher_input_ids
    ):
        student_answer_index, student_answer_size = self._get_start_and_size_answers(student_labels)
        teacher_answer_index, teacher_answer_size = self._get_start_and_size_answers(teacher_labels)

        if self.skip_student_eos:
            student_answer_size = [s - 1 for s in student_answer_size]
        if self.skip_teacher_eos:
            teacher_answer_size = [s - 1 for s in teacher_answer_size]

        if not student_answer_size or not teacher_answer_size or \
                max(max(student_answer_size), max(teacher_answer_size)) <= 0:
            return torch.zeros(1, device=student_logits.device, requires_grad=True) * student_logits.sum() * 0

        batch_size = student_logits.size(0)
        losses = []

        for i in range(batch_size):
            s_start, s_size = student_answer_index[i], student_answer_size[i]
            t_start, t_size = teacher_answer_index[i], teacher_answer_size[i]
            if s_size <= 0 or t_size <= 0:
                losses.append(student_logits[i].sum() * 0.0)
                continue

            s_ids = student_input_ids[i, s_start:s_start + s_size].tolist()
            t_ids = teacher_input_ids[i, t_start:t_start + t_size].tolist()

            eos_id_t = self.teacher_tokenizer.eos_token_id
            if eos_id_t is not None and eos_id_t in t_ids:
                t_ids = t_ids[:t_ids.index(eos_id_t)]

            eos_id_s = self.student_tokenizer.eos_token_id
            if eos_id_s is not None and eos_id_s in s_ids:
                s_ids = s_ids[:s_ids.index(eos_id_s)]

            def strip_tail_tokens(ids, tokenizer):
                if tokenizer is None:
                    return ids
                tail_ids = set()
                if tokenizer.pad_token_id is not None:
                    tail_ids.add(tokenizer.pad_token_id)
                if tokenizer.eos_token_id is not None:
                    tail_ids.add(tokenizer.eos_token_id)
                j = len(ids)
                while j > 0 and ids[j - 1] in tail_ids:
                    j -= 1
                ids = ids[:j]
                endoftext_seq = tokenizer.encode("<|endoftext|>", add_special_tokens=False)
                if endoftext_seq:
                    eot_len = len(endoftext_seq)
                    while len(ids) >= eot_len and ids[-eot_len:] == endoftext_seq:
                        ids = ids[:-eot_len]
                return ids

            s_ids = strip_tail_tokens(s_ids, self.student_tokenizer)
            t_ids = strip_tail_tokens(t_ids, self.teacher_tokenizer)

            if not s_ids or not t_ids:
                losses.append(student_logits[i].sum() * 0.0)
                continue

            s_probs = F.softmax(student_logits[i, s_start:s_start + len(s_ids)] / self.student_temperature, dim=-1)
            t_probs = F.softmax(teacher_logits[i, t_start:t_start + len(t_ids)] / self.teacher_temperature, dim=-1)

            s_groups, t_groups = self._build_alignment_groups_from_ids(s_ids, t_ids)

            if self.use_byte_loss and self.teacher_bc is not None and self.student_bc is not None:
                prefix_ids = teacher_input_ids[i, :t_start]
                prefix_str = self.teacher_tokenizer.decode(prefix_ids, skip_special_tokens=False)
                try:
                    t_sampler = self.teacher_bc.BytewiseBatchSampler(
                        self.teacher_bc, batch_size=1, filter_tensors=True, do_gc=False
                    )
                    s_sampler = self.student_bc.BytewiseBatchSampler(
                        self.student_bc, batch_size=1, filter_tensors=True, do_gc=False
                    )
                    t_sampler.add_context([prefix_str.encode('utf-8')])
                    s_sampler.add_context([prefix_str.encode('utf-8')])
                    use_byte = True
                except AssertionError:
                    t_sampler = s_sampler = None
                    use_byte = False
            else:
                t_sampler = s_sampler = None
                use_byte = False

            group_losses = []

            for gi, (s_group, t_group) in enumerate(zip(s_groups, t_groups)):
                group_start_t = t_start + t_group[0] if t_group else t_start
                group_end_t = t_start + t_group[-1] + 1 if t_group else t_start
                group_ids_t = teacher_input_ids[i, group_start_t:group_end_t]
                group_text = self.teacher_tokenizer.decode(group_ids_t, skip_special_tokens=False)
                group_bytes = group_text.encode('utf-8')

                if not s_group or not t_group:
                    if use_byte and group_bytes:
                        try:
                            t_sampler.add_context([group_bytes])
                            s_sampler.add_context([group_bytes])
                        except AssertionError:
                            use_byte = False
                    continue

                if len(s_group) == 1 and len(t_group) == 1:
                    s_prob = s_probs[s_group[0]]
                    t_prob = t_probs[t_group[0]]
                    t_id = t_ids[t_group[0]]

                    if self._vocab_mapping is not None and t_id in self._vocab_mapping:
                        loss_g = self._jsd_single_token(s_prob, t_prob)
                    else:
                        loss_g = self._uld_single_token(s_prob, t_prob)

                    if use_byte:
                        try:
                            t_sampler.add_context([group_bytes])
                            s_sampler.add_context([group_bytes])
                        except AssertionError:
                            use_byte = False

                else:
                    if use_byte:
                        dev = student_logits.device
                        try:
                            t_dists = self._compute_byte_dists_from_sampler(t_sampler, group_bytes, dev)
                            s_dists = self._compute_byte_dists_from_sampler(s_sampler, group_bytes, dev)
                            if t_dists.shape[0] > 0 and s_dists.shape[0] > 0:
                                loss_g = self._jsd_from_byte_dists(t_dists, s_dists)
                            else:
                                loss_g = student_logits[i].sum() * 0.0
                        except AssertionError:
                            use_byte = False
                            s_grp_prob = self._merge_probs_for_group(s_probs, s_group, s_ids)
                            t_grp_prob = self._merge_probs_for_group(t_probs, t_group, t_ids)
                            loss_g = self._uld_single_token(s_grp_prob, t_grp_prob)
                    else:
                        s_grp_prob = self._merge_probs_for_group(s_probs, s_group, s_ids)
                        t_grp_prob = self._merge_probs_for_group(t_probs, t_group, t_ids)
                        loss_g = self._uld_single_token(s_grp_prob, t_grp_prob)

                group_losses.append(loss_g)

            del t_sampler, s_sampler
            gc.collect()
            _empty_cache()

            if group_losses:
                loss_i = torch.stack(group_losses).mean()
            else:
                loss_i = student_logits[i].sum() * 0.0
            losses.append(loss_i)

        return self.distillation_weight * torch.stack(losses).mean()

    def _merge_probs_for_group(self, probs, group, token_ids):
        """Merge прероятностей для группы токенов через произведение маргинальных."""
        eps = 1e-8
        if len(group) == 1:
            return probs[group[0]]
        first = group[0]
        marginal = probs[first]  # (V,)
        cond = torch.ones(1, device=probs.device, dtype=probs.dtype)
        for pos in group[1:]:
            cond = cond * probs[pos, token_ids[pos]].clamp_min(eps)
        return marginal * cond

    @staticmethod
    def _get_start_and_size_answers(answer_tensors):
        indices, sizes = [], []
        for ans in answer_tensors:
            mask = ans.ne(-100)
            if not mask.any():
                indices.append(0); sizes.append(0)
                continue
            valid = mask.nonzero(as_tuple=True)[0]
            indices.append(int(valid[0].item()))
            sizes.append(int(mask.sum().item()))
        return indices, sizes


# ==================== GOLDTrainer ====================

class GOLDTrainer(Trainer):
    _tag_names = ["trl", "gold"]
    _name = "GOLD"

    def __init__(
        self,
        model: PreTrainedModel | nn.Module,
        teacher_model: PreTrainedModel | nn.Module,
        args: TrainingArguments,
        train_dataset: Dataset,
        processing_class: PreTrainedTokenizerBase,
        data_collator: DataCollator | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple = (None, None),
    ):
        if data_collator is None:
            data_collator = DefaultDataCollator()
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            callbacks=callbacks,
            optimizers=optimizers,
            processing_class=processing_class,
        )
        self.processing_class = processing_class
        self.teacher_model = teacher_model
        self.use_uld_loss = getattr(args, "use_uld_loss", False)

        self.teacher_tokenizer = None
        if self.use_uld_loss and getattr(args, "teacher_tokenizer_name_or_path", None):
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                args.teacher_tokenizer_name_or_path
            )
            if not self.teacher_tokenizer.pad_token:
                self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token

        self.lmbda = getattr(args, "lmbda", 0.5)
        self.beta = getattr(args, "beta", 0.5)
        self.temperature = getattr(args, "temperature", 0.8)
        self.top_p = getattr(args, "top_p", 0.9)
        self.num_generations = getattr(args, "num_generations", 1)

        # Счётчики для логирования
        self._on_policy_loss_total = 0.0
        self._off_policy_loss_total = 0.0
        self._on_policy_step_equiv = 0.0
        self._off_policy_step_equiv = 0.0
        self._buffered_inputs = None
        self._buffered_on_policy = None
        self._buffered_text_logs = None
        self._step = 0
        self._matched_sum = 0.0
        self._unmatched_sum = 0.0
        self._matched_step_eq = 0.0
        self._unmatched_step_eq = 0.0

        pad_token_id = self.processing_class.pad_token_id if self.processing_class.pad_token_id is not None else 0
        self.generation_config = GenerationConfig(
            max_new_tokens=getattr(args, "max_completion_length", 128),
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            top_k=getattr(args, "top_k", 50),
            pad_token_id=pad_token_id,
        )

        self.log_completions = getattr(args, "log_completions", False)
        self.log_completion_steps = getattr(args, "log_completion_steps", 100)
        self.num_completions_to_print = getattr(args, "num_completions_to_print", 2)
        self._textual_logs = {"prompt": deque(maxlen=100), "completion": deque(maxlen=100)}

        # ИСПРАВЛЕНИЕ #2: передаём готовые модели, не пути
        self.uld_loss_fn = ULDLoss(
            config=args,
            student_tokenizer=self.processing_class,
            teacher_tokenizer=self.teacher_tokenizer,
            device=self.accelerator.device,
            teacher_model=teacher_model if getattr(args, "use_byte_loss", False) else None,
            student_model=model if getattr(args, "use_byte_loss", False) else None,
        )

        # В GOLDTrainer.__init__ после создания uld_loss_fn:
        if getattr(args, "use_byte_loss", False):
            teacher_dev = self.uld_loss_fn.teacher_bc.device
            student_dev = self.uld_loss_fn.student_bc.device
            expected = self.accelerator.device
            def _same_device(a, b):
                a, b = torch.device(a), torch.device(b)
                return a.type == b.type and (a.index or 0) == (b.index or 0)

            assert _same_device(teacher_dev, expected), \
                f"teacher_bc.device={teacher_dev} != accelerator.device={expected}"
            assert _same_device(student_dev, expected), \
                f"student_bc.device={student_dev} != accelerator.device={expected}"
            
            print(f"[byte_loss] bc devices OK: teacher={teacher_dev}, student={student_dev}")

    def _get_train_sampler(self, dataset=None):
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.per_device_train_batch_size * self.accelerator.num_processes,
            repeat_count=self.args.gradient_accumulation_steps,
            shuffle=True,
            seed=self.args.seed,
        )

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.gradient_accumulation_steps,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            )
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _prepare_inputs(self, generation_batch):
        if not self.model.training:
            return generation_batch
        buffer_steps = self.args.gradient_accumulation_steps
        if self._step % buffer_steps == 0 or self._buffered_inputs is None:
            self._fill_buffer(generation_batch, buffer_steps)
        slice_idx = self._step % buffer_steps
        inputs = self._buffered_inputs[slice_idx]
        self._step += 1
        return inputs

    def _fill_buffer(self, generation_batch, buffer_steps):
        slices = split_tensor_dict(generation_batch, buffer_steps)
        if self.accelerator.is_main_process:
            on_policy_flags = [random.random() <= self.lmbda for _ in range(buffer_steps)]
        else:
            on_policy_flags = [False] * buffer_steps
        on_policy_flags = broadcast_object_list(on_policy_flags, from_process=0)
        on_policy_indices = [i for i, flag in enumerate(on_policy_flags) if flag]

        self._buffered_inputs = [None] * buffer_steps
        self._buffered_on_policy = on_policy_flags
        self._buffered_text_logs = [None] * buffer_steps

        for i, flag in enumerate(on_policy_flags):
            if not flag:
                self._buffered_inputs[i] = slices[i]

        if on_policy_indices:
            self._generate_on_policy_for_slices(slices, on_policy_indices)

    def _generate_on_policy_for_slices(self, slices, on_policy_indices):
        prompt_ids_list = []
        local_slice_indices = []
        for slice_idx in on_policy_indices:
            slice_inputs = slices[slice_idx]
            prompt_attention_mask = slice_inputs.get("prompt_attention_mask")
            for prompt_idx, prompt in enumerate(slice_inputs["prompts"]):
                if prompt_attention_mask is not None:
                    prompt = prompt[prompt_attention_mask[prompt_idx].bool()]
                prompt_ids_list.append(prompt.tolist())
                local_slice_indices.append(slice_idx)

        prompts_text = self.processing_class.batch_decode(prompt_ids_list, skip_special_tokens=True)
        prompts_text_special = self.processing_class.batch_decode(prompt_ids_list, skip_special_tokens=False)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        all_completion_ids = []

        # Батчевая генерация вместо цикла по одному примеру
        # ОПТИМИЗАЦИЯ: генерируем батчами, а не по одному
        device = self.accelerator.device
        BATCH = max(1, 4)  # можно настроить под GPU
        for batch_start in range(0, len(prompt_ids_list), BATCH):
            batch_pids = prompt_ids_list[batch_start:batch_start + BATCH]
            max_p_len = max(len(p) for p in batch_pids)
            pad_id = self.processing_class.pad_token_id or 0
            input_tensors = torch.stack([
                F.pad(torch.tensor(p, device=device, dtype=torch.long),
                      (max_p_len - len(p), 0), value=pad_id)
                for p in batch_pids
            ])
            attn_masks = (input_tensors != pad_id).long()
            with torch.no_grad():
                outputs = unwrapped_model.generate(
                    input_tensors,
                    attention_mask=attn_masks,
                    max_new_tokens=self.generation_config.max_new_tokens,
                    temperature=self.generation_config.temperature,
                    top_p=self.generation_config.top_p,
                    do_sample=True,
                    pad_token_id=self.processing_class.pad_token_id,
                    eos_token_id=self.processing_class.eos_token_id,
                )
            for j, (p, out) in enumerate(zip(batch_pids, outputs)):
                all_completion_ids.append(out[max_p_len:].tolist())

        self._process_completions_to_buffer(
            slices, on_policy_indices, local_slice_indices, all_completion_ids,
            prompt_ids_list, prompts_text_special, prompts_text,
            self.generation_config.max_new_tokens,
        )

    def _process_completions_to_buffer(
        self, slices, on_policy_indices, local_slice_indices, completion_ids,
        prompt_ids_list, prompts_text_special, prompts_text, max_comp_len
    ):
        device = self.accelerator.device
        pad_token_id = self.processing_class.pad_token_id or 0

        slice_comp = {idx: [] for idx in on_policy_indices}
        slice_pids = {idx: [] for idx in on_policy_indices}
        slice_pspecial = {idx: [] for idx in on_policy_indices}
        slice_ptext = {idx: [] for idx in on_policy_indices}

        for i, sidx in enumerate(local_slice_indices):
            slice_comp[sidx].append(completion_ids[i])
            slice_pids[sidx].append(prompt_ids_list[i])
            slice_pspecial[sidx].append(prompts_text_special[i])
            slice_ptext[sidx].append(prompts_text[i])

        for sidx in on_policy_indices:
            slice_inputs = slices[sidx]
            comps = slice_comp[sidx]
            prompts = slice_pids[sidx]
            pspecial = slice_pspecial[sidx]
            ptext = slice_ptext[sidx]
            prompt_max_len = (
                self.args.max_length - max_comp_len if getattr(self.args, "max_length", None) else None
            )
            truncated_prompts = [
                torch.tensor(p[:prompt_max_len] if prompt_max_len and len(p) > prompt_max_len else p,
                              device=device, dtype=torch.long)
                for p in prompts
            ]
            prompt_ids_pad = pad(truncated_prompts, padding_side="left", padding_value=pad_token_id)
            prompt_attn_pad = pad(
                [torch.ones(len(p), device=device, dtype=torch.long) for p in truncated_prompts],
                padding_side="left", padding_value=0
            )

            def truncate_at_eos(ids, eos_id):
                if ids is None:
                    return []
                if eos_id is not None and eos_id in ids:
                    ids = ids[:ids.index(eos_id) + 1]  # включаем EOS
                return ids[:max_comp_len]
                
            eos_id = self.processing_class.eos_token_id
            comp_tensors = [
                torch.tensor(truncate_at_eos(ids, eos_id), device=device, dtype=torch.long)
                for ids in comps
            ]

            comp_texts = [
                self.processing_class.decode(truncate_at_eos(ids, eos_id), skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
                for ids in comps
            ]

            if comp_tensors:
                comp_ids_pad = pad(comp_tensors, padding_side="right", padding_value=pad_token_id)
                comp_attn_pad = pad(
                    [torch.ones(t.size(0), device=device, dtype=torch.long) for t in comp_tensors],
                    padding_side="right", padding_value=0
                )
            else:
                comp_ids_pad = torch.empty((0, 0), device=device, dtype=torch.long)
                comp_attn_pad = torch.empty((0, 0), device=device, dtype=torch.long)

            new_input_ids = torch.cat([prompt_ids_pad, comp_ids_pad], dim=1)
            new_attention_mask = torch.cat([prompt_attn_pad, comp_attn_pad], dim=1)
            prompt_lengths = torch.full((prompt_ids_pad.shape[0],), prompt_ids_pad.shape[1], device=device)
            positions = torch.arange(new_input_ids.shape[1], device=device).unsqueeze(0)
            comp_mask = positions >= prompt_lengths.unsqueeze(1)
            labels = torch.full_like(new_input_ids, -100)
            labels[comp_mask & new_attention_mask.bool()] = new_input_ids[comp_mask & new_attention_mask.bool()]
            if pad_token_id is not None:
                labels[new_input_ids == pad_token_id] = -100

            updated = dict(slice_inputs)
            updated.update({
                "input_ids": new_input_ids,
                "attention_mask": new_attention_mask,
                "labels": labels,
                "original_prompt_text": pspecial,
                "original_completion_text": comp_texts,
            })
            self._buffered_inputs[sidx] = updated
            self._buffered_text_logs[sidx] = (ptext, comp_texts)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.use_uld_loss and self.teacher_tokenizer is not None:
            prompt_texts = inputs["original_prompt_text"]
            completion_texts = inputs["original_completion_text"]
            teacher_input_ids, teacher_labels, teacher_attention_mask, _ = build_teacher_inputs_from_texts(
                self.teacher_tokenizer, prompt_texts, completion_texts,
                device=self.accelerator.device,
                max_completion_tokens=getattr(self.args, "max_completion_length", 256) * 2, # умножаем на 2 — учитель токенизирует мелче чем студент
                )

            outputs_student = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False
            )
            self.teacher_model.eval()
            with torch.no_grad():
                outputs_teacher = self.teacher_model(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask
                )

            student_labels = inputs["labels"].clone()
            # if self.processing_class.pad_token_id is not None:
            #     student_labels[student_labels == self.processing_class.pad_token_id] = -100
            # if self.teacher_tokenizer.pad_token_id is not None:
            #     teacher_labels[teacher_labels == self.teacher_tokenizer.pad_token_id] = -100

            # ДОБАВИТЬ сразу перед вызовом self.uld_loss_fn(...)
            teacher_valid = (teacher_labels != -100).sum(dim=-1)
            student_valid = (student_labels != -100).sum(dim=-1)
            # print(f"[compute_loss] teacher_labels valid tokens per row: {teacher_valid.tolist()}", flush=True)
            # print(f"[compute_loss] student_labels valid tokens per row: {student_valid.tolist()}", flush=True)
            # print(f"[compute_loss] teacher_input_ids shape: {teacher_input_ids.shape}", flush=True)
            # Покажем последние 10 токенов учителя чтобы понять где заканчивается реальный контент
            # print(f"[compute_loss] teacher_input_ids[-10:]: {teacher_input_ids[0, -10:].tolist()}", flush=True)
            # print(f"[compute_loss] teacher_labels[-10:]: {teacher_labels[0, -10:].tolist()}", flush=True)
            # print(f"[compute_loss] teacher eos_token_id={self.teacher_tokenizer.eos_token_id}, pad_token_id={self.teacher_tokenizer.pad_token_id}", flush=True)

            loss = self.uld_loss_fn(
                student_logits=outputs_student.logits,
                teacher_logits=outputs_teacher.logits,
                student_labels=student_labels,
                teacher_labels=teacher_labels,
                student_input_ids=inputs["input_ids"],
                teacher_input_ids=teacher_input_ids,
            )

            # ИСПРАВЛЕНИЕ #6: освобождаем память учителя
            del outputs_teacher
            _empty_cache()

        else:
            outputs_student = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            self.teacher_model.eval()
            with torch.no_grad():
                outputs_teacher = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
            prompt_lengths = inputs["prompts"].shape[1]
            shifted_student = outputs_student.logits[:, prompt_lengths - 1:-1]
            shifted_teacher = outputs_teacher.logits[:, prompt_lengths - 1:-1]
            shifted_labels = inputs["labels"][:, prompt_lengths:]
            loss = self.generalized_jsd_loss(
                student_logits=shifted_student,
                teacher_logits=shifted_teacher,
                labels=shifted_labels,
                beta=self.beta,
                temperature=self.temperature,
            )
            del outputs_teacher
            _empty_cache()

        return (loss, outputs_student) if return_outputs else loss

    @staticmethod
    def generalized_jsd_loss(
        student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0,
        reduction="batchmean", logits_are_probs=False
    ):
        if logits_are_probs:
            student_log_probs = torch.log(student_logits.clamp_min(1e-8))
            teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
        else:
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture = torch.logsumexp(
                torch.stack([
                    student_log_probs + torch.log1p(-beta_t),
                    teacher_log_probs + torch.log(beta_t)
                ]), dim=0
            )
            kl_student = F.kl_div(mixture, student_log_probs, reduction="none", log_target=True)
            kl_teacher = F.kl_div(mixture, teacher_log_probs, reduction="none", log_target=True)
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        if labels is not None:
            mask = labels != -100
            if mask.any():
                jsd = jsd[mask].sum(dim=-1)
            else:
                return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        else:
            jsd = jsd.sum(dim=-1)

        if reduction == "batchmean":
            return jsd.mean()
        elif reduction == "sum":
            return jsd.sum()
        return jsd


def _empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()