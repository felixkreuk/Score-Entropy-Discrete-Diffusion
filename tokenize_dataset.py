from datasets import load_dataset, concatenate_datasets
from transformers import GPT2TokenizerFast
import os
import pickle
from tqdm import trange
import errno
from time import sleep


def create_lock_file(lock_file):
    try:
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL)
        # Lock file created successfully, you can now write to it if needed
        os.close(fd)
        return True
    except OSError as e:
        if e.errno == errno.EEXIST:
            # print(f"Lock file {lock_file} already exists.")
            return False
        else:
            # An unexpected error occurred
            raise


def process_dataset(dataset_dict: dict, map_function: callable, K: int, output_dir: str, **kwargs):
   # Create output directory if it doesn't exist
   if not os.path.exists(output_dir):
      os.makedirs(output_dir, exist_ok=True)

   processed_datasets = {}

   if not isinstance(dataset_dict, dict):
      dataset_dict = {"data": dataset_dict}

   for dataset_name, dataset in dataset_dict.items():
      processed_chunks = []

      for i in trange(K):
         # Check if this chunk has already been processed
         chunk_file = os.path.join(output_dir, f'{dataset_name}_chunk_{i}.pickle')
         lock_file = chunk_file + ".lock"

         if os.path.exists(chunk_file):
            continue

         if create_lock_file(lock_file):
            print(f"{chunk_file} computing")
            # Shard the dataset and process the chunk
            chunk = dataset.shard(K, i)
            processed_chunk = chunk.map(map_function, batched=True, **kwargs)
            # Save the processed chunk to disk
            with open(chunk_file, 'wb') as f:
               pickle.dump(processed_chunk, f)
            os.remove(lock_file)
         else:
            continue
      print("launched all shards")

      while True:
         exist = 0
         for i in range(K):
            chunk_file = os.path.join(output_dir, f'{dataset_name}_chunk_{i}.pickle')
            if os.path.exists(chunk_file):
               exist += 1
         if exist == K:
            break
         print(f"waiting ({exist}/{K})")
         sleep(1)
      print("finished waiting")

      for i in trange(K):
         # Load the processed chunk from disk
         with open(chunk_file, 'rb') as f:
            processed_chunk = pickle.load(f)
         processed_chunks.append(processed_chunk)

      # Merge the processed chunks into a single dataset
      print("merging chunks...")
      processed_datasets[dataset_name] = concatenate_datasets(processed_chunks)
      print(processed_datasets[dataset_name])
      print(f"{dataset_name} merged")

   return processed_datasets


def main():
   dataset = load_dataset("/fsx-labs/broz/data/shuffled/c4", cache_dir="/fsx-codegen/felixkreuk/datasets/c4")
   detokenizer = None

   def _apply_detokenizer(detokenizer):
      def detok(text):
         for i, t in enumerate(text, 0):
               text[i] = detokenizer(t)
         return text
      return detok

   tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
   EOS = tokenizer.encode(tokenizer.eos_token)[0]

   def preprocess_and_tokenize(example):
      text = example["text"]
      # print(list(example.keys()))
      # exit()

      if detokenizer is not None:
         text = _apply_detokenizer(detokenizer)(text)

      tokens = tokenizer(text, return_attention_mask=False)
      # add in EOS token following
      # https://github.com/jcpeterson/openwebtext/blob/master/tokenize_text.py#L67
      for token in tokens['input_ids']:
         token.append(EOS)
      return tokens

   dataset = load_dataset("/fsx-labs/broz/data/shuffled/c4", cache_dir="/fsx-codegen/felixkreuk/datasets/c4")
   process_dataset(dataset, preprocess_and_tokenize, 1000, "/fsx-codegen/felixkreuk/datasets/c4/tokenized")

if __name__ == "__main__":
   main()
