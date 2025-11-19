"""
embeddings.py

Быстрое извлечение эмбеддингов:
- deepvk/USER-bge-m3
- cointegrated/rubert-tiny2
- cointegrated/LaBSE-en-ru
- ai-forever/ru-en-RoSBERTa
- google/siglip2-so400m-patch14-384 (текст + картинки)

Все функции возвращают torch.Tensor [n_samples, dim] на CPU.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Iterable, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from PIL import Image


TextLike = Union[str, Iterable[str]]
PathLike = Union[str, Iterable[str]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_list_text(texts: TextLike) -> List[str]:
    if isinstance(texts, str):
        return [texts]
    return list(texts)


def _ensure_list_paths(paths: PathLike) -> List[str]:
    if isinstance(paths, str):
        return [paths]
    return list(paths)


def _get_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _batch_iter(items: List, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# ---------------------------------------------------------------------------
# deepvk/USER-bge-m3  (CLS-пулинг + L2-норма)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_user_bge_m3():
    tokenizer = AutoTokenizer.from_pretrained("deepvk/USER-bge-m3")
    model = AutoModel.from_pretrained("deepvk/USER-bge-m3")
    model.eval()
    return tokenizer, model


def embed_user_bge_m3(
    texts: TextLike,
    device: str | torch.device | None = None,
    batch_size: int = 32,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Эмбеддинги deepvk/USER-bge-m3 (1024 dim, CLS-пулинг).
    """
    texts = _ensure_list_text(texts)
    tokenizer, model = _load_user_bge_m3()
    device = _get_device(device)
    model.to(device)

    all_embs = []

    for batch in _batch_iter(texts, batch_size):
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            # CLS-пулинг
            embs = outputs.last_hidden_state[:, 0, :]  # [B, D]
            if normalize:
                embs = F.normalize(embs, p=2, dim=1)

        all_embs.append(embs.cpu())

    return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# cointegrated/rubert-tiny2  (CLS-пулинг + L2-норма)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_rubert_tiny2():
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
    model.eval()
    return tokenizer, model


def embed_rubert_tiny2(
    texts: TextLike,
    device: str | torch.device | None = None,
    batch_size: int = 32,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Эмбеддинги cointegrated/rubert-tiny2 (312 dim, CLS-пулинг).
    """
    texts = _ensure_list_text(texts)
    tokenizer, model = _load_rubert_tiny2()
    device = _get_device(device)
    model.to(device)

    all_embs = []

    for batch in _batch_iter(texts, batch_size):
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            embs = outputs.last_hidden_state[:, 0, :]  # CLS
            if normalize:
                embs = F.normalize(embs, p=2, dim=1)

        all_embs.append(embs.cpu())

    return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# cointegrated/LaBSE-en-ru  (pooler_output + L2-норма)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_labse_en_ru():
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
    model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")
    model.eval()
    return tokenizer, model


def embed_labse_en_ru(
    texts: TextLike,
    device: str | torch.device | None = None,
    batch_size: int = 32,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Эмбеддинги cointegrated/LaBSE-en-ru (pooler_output).
    """
    texts = _ensure_list_text(texts)
    tokenizer, model = _load_labse_en_ru()
    device = _get_device(device)
    model.to(device)

    all_embs = []

    for batch in _batch_iter(texts, batch_size):
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            embs = outputs.pooler_output  # [B, D]
            if normalize:
                embs = F.normalize(embs, p=2, dim=1)

        all_embs.append(embs.cpu())

    return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# ai-forever/ru-en-RoSBERTa  (CLS/mean-пулинг + L2-норма)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_rosberta():
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/ru-en-RoSBERTa")
    model = AutoModel.from_pretrained("ai-forever/ru-en-RoSBERTa")
    model.eval()
    return tokenizer, model


def _pool_hidden(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    method: str = "cls",
) -> torch.Tensor:
    """
    Пулинг как в модельной карточке ru-en-RoSBERTa.
    hidden_state: [B, T, D]
    attention_mask: [B, T]
    """
    if method == "cls":
        return hidden_state[:, 0, :]
    elif method == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def embed_rosberta(
    texts: TextLike,
    device: str | torch.device | None = None,
    batch_size: int = 32,
    pooling: str = "cls",  # "cls" или "mean"
    normalize: bool = True,
) -> torch.Tensor:
    """
    Эмбеддинги ai-forever/ru-en-RoSBERTa.

    По-хорошему для этого энкодера тексты лучше подавать уже с префиксами:
    - "classification: ..."
    - "clustering: ..."
    - "search_query: ..." / "search_document: ..."
    """
    texts = _ensure_list_text(texts)
    tokenizer, model = _load_rosberta()
    device = _get_device(device)
    model.to(device)

    all_embs = []

    for batch in _batch_iter(texts, batch_size):
        enc = tokenizer(
            batch,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            embs = _pool_hidden(
                outputs.last_hidden_state,
                enc["attention_mask"],
                method=pooling,
            )
            if normalize:
                embs = F.normalize(embs, p=2, dim=1)

        all_embs.append(embs.cpu())

    return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# google/siglip2-so400m-patch14-384  (текст + картинки)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_siglip2():
    """
    Общий модель + процессор для текста и картинок.
    """
    processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")
    model = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384")
    model.eval()
    return processor, model


def embed_siglip2_text(
    texts: TextLike,
    device: str | torch.device | None = None,
    batch_size: int = 32,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Текстовые эмбеддинги SigLIP2 (общий мультимодальный space).
    """
    texts = _ensure_list_text(texts)
    processor, model = _load_siglip2()
    device = _get_device(device)
    model.to(device)

    all_embs = []

    for batch in _batch_iter(texts, batch_size):
        enc = processor(
            text=batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            embs = model.get_text_features(**enc)  # [B, D]
            if normalize:
                embs = F.normalize(embs, p=2, dim=1)

        all_embs.append(embs.cpu())

    return torch.cat(all_embs, dim=0)


def embed_siglip2_images(
    image_paths: PathLike,
    device: str | torch.device | None = None,
    batch_size: int = 16,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Эмбеддинги картинок SigLIP2.

    Ожидает пути к файлам. Если нужно работать с уже загруженными PIL.Image,
    можно слегка адаптировать функцию (убрать open()).
    """
    paths = _ensure_list_paths(image_paths)
    processor, model = _load_siglip2()
    device = _get_device(device)
    model.to(device)

    all_embs = []

    for batch_paths in _batch_iter(paths, batch_size):
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        enc = processor(
            images=images,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            embs = model.get_image_features(**enc)  # [B, D]
            if normalize:
                embs = F.normalize(embs, p=2, dim=1)

        all_embs.append(embs.cpu())

    return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# Пример использования
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    texts = ["Привет мир", "Как дела?"]
    img_paths = ["example1.jpg", "example2.jpg"]

    print("USER-bge-m3:", embed_user_bge_m3(texts).shape)
    print("rubert-tiny2:", embed_rubert_tiny2(texts).shape)
    print("LaBSE-en-ru:", embed_labse_en_ru(texts).shape)
    print("RoSBERTa:", embed_rosberta(texts).shape)

    print("SigLIP2 text:", embed_siglip2_text(texts).shape)
    print("SigLIP2 images:", embed_siglip2_images(img_paths).shape)