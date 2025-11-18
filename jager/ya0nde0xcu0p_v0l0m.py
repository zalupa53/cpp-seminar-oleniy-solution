import json
import pickle
import io
import os
import re
from typing import List, Dict

import torch
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from rich.progress import track
from rich.console import Console

from utils import construct_messages, prepare_inputs_for_vllm

# Set environment variables
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = "spawn"

console = Console()

# Batch size for processing
BATCH_SIZE = 16


def create_thinking_prompt(question_text: str) -> str:
    """
    Создаёт промпт с системной стратегией размышлений для анализа каждого утверждения
    """
    prompt = f"""You are an expert in mathematics, physics, and geometry. You will be given an image with a diagram or graph and several statements labeled A, B, C, D, E, etc.

Your task is to analyze the image carefully and determine which statements are TRUE.

IMPORTANT STRATEGY - THINK STEP BY STEP:

1. OBSERVE THE IMAGE:
   - Identify all geometric elements (points, lines, angles, shapes)
   - Note all measurements, labels, and annotations
   - Identify any special markings (equal segments, right angles, parallel lines)
   - For physics: identify objects, forces, directions, states

2. ANALYZE EACH STATEMENT INDEPENDENTLY:
   - Extract what each statement claims
   - Check if the claim can be verified from the image
   - Look for evidence supporting OR contradicting the statement
   - Consider mathematical relationships and physical laws

3. VERIFY SYSTEMATICALLY:
   - For angles: check marked angles, use angle relationships (vertical, supplementary, etc.)
   - For lengths: check marked equal segments, use geometric theorems
   - For physics: apply physical principles (equilibrium, buoyancy, pressure, etc.)
   - Calculate when necessary to confirm numerical values

4. MAKE YOUR DECISION:
   - A statement is TRUE only if it is directly supported by the image
   - If you cannot verify a statement from the image, it is FALSE
   - Multiple statements can be true simultaneously

Here are the statements to analyze:

{question_text}

RESPONSE FORMAT:
First, think through each statement step by step in your analysis.
Then, on the last line, output ONLY the letters of TRUE statements as a single string (e.g., "A", "BC", "ADE").

Begin your analysis:"""
    
    return prompt


def extract_answer_from_thinking(response: str) -> str:
    """
    Извлекает финальный ответ из ответа модели с размышлениями
    """
    # Ищем последнюю строку с буквами A-F
    lines = response.strip().split('\n')
    
    # Ищем с конца файла паттерн с ответом
    for line in reversed(lines):
        line = line.strip()
        # Ищем строку, которая содержит только буквы A-F (возможно с пробелами/запятыми)
        clean_line = re.sub(r'[^A-F]', '', line.upper())
        if clean_line and len(clean_line) <= 6:  # Максимум 6 вариантов A-F
            # Сортируем буквы в алфавитном порядке для консистентности
            return ''.join(clean_line)
    
    # Если не нашли явный ответ, пытаемся найти упоминания "Answer:", "Final answer:", etc.
    for line in reversed(lines):
        if any(keyword in line.lower() for keyword in ['answer:', 'final:', 'result:', 'conclusion:']):
            clean_line = re.sub(r'[^A-F]', '', line.upper())
            if clean_line:
                return ''.join(clean_line)
    
    # В крайнем случае возвращаем все найденные буквы из последних 3 строк
    last_text = ' '.join(lines[-3:])
    clean_answer = re.sub(r'[^A-F]', '', last_text.upper())
    if clean_answer:
        return ''.join(clean_line)
    
    return ""


def main():
    # ====== КОНФИГУРАЦИЯ МОДЕЛИ ======
    # Выберите одну из конфигураций:
    
    # ВАРИАНТ 1: Используем текущую Qwen модель (рекомендуется если уже загружена)
    checkpoint_path = "./Qwen3-VL-8B-Thinking-AWQ-4bit"
    max_model_len = 12288
    batch_size = 16
    
    # ВАРИАНТ 2: Qwen2-VL-7B для максимального качества
    # checkpoint_path = "Qwen/Qwen2-VL-7B-Instruct"
    # max_model_len = 16384
    # batch_size = 6
    
    # ВАРИАНТ 3: Qwen2-VL-2B для максимальной скорости
    # checkpoint_path = "Qwen/Qwen2-VL-2B-Instruct"
    # max_model_len = 8192
    # batch_size = 16
    
    # ВАРИАНТ 4: InternVL2-8B для геометрии
    # checkpoint_path = "OpenGVLab/InternVL2-8B"
    # max_model_len = 12288
    # batch_size = 6
    
    console.log(f"Loading processor from {checkpoint_path}...")
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    
    console.log(f"Loading model from {checkpoint_path}...")
    llm = LLM(
        model=checkpoint_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.97,
        enforce_eager=False,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=max_model_len,
        seed=42
    )
    
    # Параметры для thinking - оптимизированы для точности
    sampling_params = SamplingParams(
        temperature=0.4,  # Низкая температура для точных ответов
        max_tokens=16384,  # Достаточно для развёрнутых размышлений
        top_k=30,
        top_p=0.85,
        repetition_penalty=1.05,
        stop_token_ids=[],
    )
    
    # Обновляем глобальную переменную batch_size
    global BATCH_SIZE
    BATCH_SIZE = batch_size
    
    
    # Load input data
    console.log("Loading input data...")
    with open('input.pickle', "rb") as input_file:
        model_input = pickle.load(input_file)
    
    # Prepare batches
    console.log(f"Processing {len(model_input)} rows in batches of {BATCH_SIZE}...")
    model_output = []
    
    # Save temporary images
    temp_images = {}
    console.log("Preparing images...")
    for row in track(model_input, description="Saving temporary images"):
        rid = row["rid"]
        image_bytes = row["image"]
        image = Image.open(io.BytesIO(image_bytes))
        temp_image_path = f"temp_image_{rid}.png"
        image.save(temp_image_path)
        temp_images[rid] = temp_image_path
    
    # Process in batches
    num_batches = (len(model_input) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(model_input))
        batch = model_input[start_idx:end_idx]
        
        console.log(f"Processing batch {batch_idx + 1}/{num_batches} (items {start_idx + 1}-{end_idx})")
        
        # Prepare batch inputs
        batch_inputs = []
        batch_rids = []
        
        for row in batch:
            rid = row["rid"]
            question_text = row["question"]
            temp_image_path = temp_images[rid]
            
            # Create thinking prompt
            formatted_question = create_thinking_prompt(question_text)
            
            # Construct messages and prepare inputs
            messages = construct_messages(temp_image_path, formatted_question)
            inputs = prepare_inputs_for_vllm(messages, processor)
            
            batch_inputs.append(inputs)
            batch_rids.append(rid)
        
        # Generate predictions for the batch
        try:
            outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
            
            # Process batch outputs
            for rid, output in zip(batch_rids, outputs):
                response = output.outputs[0].text
                
                # Extract final answer from thinking response
                prediction = extract_answer_from_thinking(response)
                
                model_output.append({"rid": rid, "answer": prediction})
                
                console.log(f"[{rid}] Answer: {prediction}")
                # Опционально: показываем краткую версию размышлений
                if len(response) > 200:
                    console.log(f"[{rid}] Thinking preview: {response[:200]}...", style="dim")
        
        except Exception as e:
            console.log(f"Error processing batch {batch_idx + 1}: {e}", style="bold red")
            # В случае ошибки пробуем обработать по одному
            for row, rid in zip(batch, batch_rids):
                try:
                    question_text = row["question"]
                    temp_image_path = temp_images[rid]
                    formatted_question = create_thinking_prompt(question_text)
                    messages = construct_messages(temp_image_path, formatted_question)
                    inputs = prepare_inputs_for_vllm(messages, processor)
                    
                    output = llm.generate([inputs], sampling_params=sampling_params)
                    response = output[0].outputs[0].text
                    prediction = extract_answer_from_thinking(response)
                    
                    model_output.append({"rid": rid, "answer": prediction})
                    console.log(f"[{rid}] Answer (retry): {prediction}")
                except Exception as e2:
                    console.log(f"[{rid}] Failed: {e2}", style="bold red")
                    model_output.append({"rid": rid, "answer": ""})
    
    # Clean up temporary files
    console.log("Cleaning up temporary files...")
    for temp_path in temp_images.values():
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # Save output
    console.log("Saving output...")
    with open('output.json', 'w') as output_file:
        json.dump(model_output, output_file, ensure_ascii=False, indent=2)
    
    console.log("Done!")


if __name__ == "__main__":
    main()