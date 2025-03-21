import argparse
import jsonlines
import torch
from glob import glob
from PIL import Image
from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

def init(model_name):
    processor = LlavaNextProcessor.from_pretrained(model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")

    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "Provide a short, one-sentence descriptive fact about this image."},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return processor, model, prompt

def generate(processor, model, prompt, image):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=100, num_return_sequences=5, num_beams=5, num_beam_groups=5, diversity_penalty=1.)

    facts = []
    for f in processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True):
        facts.append(f)

    return facts

def main(args):
    processor, model, prompt = init(args.model_name)
    results = {}

    for image_path in tqdm(glob(f"{args.image_folder}/**")):
        image = Image.open(image_path)
        facts = generate(processor, model, prompt, image)

        image_name = image_path.split('/')[-1]
        results[image_name] = facts

    with jsonlines.open(args.facts_file, mode="w") as writer:
        for name, facts in results.items():
            writer.write({name: facts})
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str)
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--facts_file", type=str, default="facts.jsonl")
    args = parser.parse_args()

    main(args)
