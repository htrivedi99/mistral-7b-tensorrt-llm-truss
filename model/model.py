import subprocess
subprocess.run(["pip", "install", "tensorrt_llm", "-U", "--pre", "--extra-index-url", "https://pypi.nvidia.com"])

import torch
from model.utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                   load_tokenizer, read_model_name, throttle_generator)

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.runtime import ModelRunnerCpp, ModelRunner
from huggingface_hub import snapshot_download

STOP_WORDS_LIST = None
BAD_WORDS_LIST = None
PROMPT_TEMPLATE = None

class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.tokenizer = None
        self.pad_id = None
        self.end_id = None
        self.runtime_rank = None
        self._data_dir = kwargs["data_dir"]

    def load(self):
        snapshot_download(
            "htrivedi99/mistral-7b-v0.2-trtllm",
            local_dir=self._data_dir,
            max_workers=4,
        )

        self.runtime_rank = tensorrt_llm.mpi_rank()

        model_name, model_version = read_model_name(f"{self._data_dir}/compiled-model")
        tokenizer_dir = "mistralai/Mistral-7B-Instruct-v0.2"

        self.tokenizer, self.pad_id, self.end_id = load_tokenizer(
            tokenizer_dir=tokenizer_dir,
            vocab_file=None,
            model_name=model_name,
            model_version=model_version,
            tokenizer_type="llama",
        )


        runner_cls = ModelRunner
        # runner_kwargs = dict(engine_dir=f"{self._data_dir}/compiled-model",
        #                      lora_dir=None,
        #                      rank=self.runtime_rank,
        #                      debug_mode=False,
        #                      lora_ckpt_source="hf",
        #                      max_batch_size=1,
        #                      max_input_len=1024,
        #                      max_output_len=2048,
        #                      max_beam_width=1,
        #                      max_attention_window_size=4096,
        #                      sink_token_length=None,
        #                      )

        runner_kwargs = dict(engine_dir=f"{self._data_dir}/compiled-model",
                             lora_dir=None,
                             rank=self.runtime_rank,
                             debug_mode=False,
                             lora_ckpt_source="hf",
                            )

        self.model = runner_cls.from_dir(**runner_kwargs)

    def parse_input(self,
                    tokenizer,
                    input_text=None,
                    prompt_template=None,
                    input_file=None,
                    add_special_tokens=True,
                    max_input_length=923,
                    pad_id=None,
                    num_prepend_vtokens=[],
                    model_name=None,
                    model_version=None):
        if pad_id is None:
            pad_id = tokenizer.pad_token_id

        batch_input_ids = []
        if input_file is None:
            for curr_text in input_text:
                if prompt_template is not None:
                    curr_text = prompt_template.format(input_text=curr_text)
                input_ids = tokenizer.encode(curr_text,
                                             add_special_tokens=add_special_tokens,
                                             truncation=True,
                                             max_length=max_input_length)
                batch_input_ids.append(input_ids)

        batch_input_ids = [
            torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
        ]
        return batch_input_ids

    def predict(self, request):

        prompt = request.pop("prompt")
        max_new_tokens = request.pop("max_new_tokens", 2048)
        temperature = request.pop("temperature", 0.9)
        top_k = request.pop("top_k",1)
        top_p = request.pop("top_p", 0)
        streaming = request.pop("streaming", False)
        streaming_interval = request.pop("streaming_interval", 3)

        batch_input_ids = self.parse_input(tokenizer=self.tokenizer,
                                      input_text=[prompt],
                                      prompt_template=None,
                                      input_file=None,
                                      add_special_tokens=None,
                                      max_input_length=1028,
                                      pad_id=self.pad_id,
                                      )
        input_lengths = [x.size(0) for x in batch_input_ids]

        outputs = self.model.generate(
            batch_input_ids,
            max_new_tokens=max_new_tokens,
            max_attention_window_size=None,
            sink_token_length=None,
            end_id=self.end_id,
            pad_id=self.pad_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=1,
            length_penalty=1,
            repetition_penalty=1,
            presence_penalty=0,
            frequency_penalty=0,
            stop_words_list=STOP_WORDS_LIST,
            bad_words_list=BAD_WORDS_LIST,
            lora_uids=None,
            streaming=streaming,
            output_sequence_lengths=True,
            return_dict=True)

        if streaming:
            streamer = throttle_generator(outputs, streaming_interval)

            def generator():
                total_output = ""
                for curr_outputs in streamer:
                    if self.runtime_rank == 0:
                        output_ids = curr_outputs['output_ids']
                        sequence_lengths = curr_outputs['sequence_lengths']
                        batch_size, num_beams, _ = output_ids.size()
                        for batch_idx in range(batch_size):
                            for beam in range(num_beams):
                                output_begin = input_lengths[batch_idx]
                                output_end = sequence_lengths[batch_idx][beam]
                                outputs = output_ids[batch_idx][beam][
                                          output_begin:output_end].tolist()
                                output_text = self.tokenizer.decode(outputs)

                                current_length = len(total_output)
                                total_output = output_text
                                yield total_output[current_length:]
            return generator()
        else:
            if self.runtime_rank == 0:
                output_ids = outputs['output_ids']
                sequence_lengths = outputs['sequence_lengths']
                batch_size, num_beams, _ = output_ids.size()
                for batch_idx in range(batch_size):
                    for beam in range(num_beams):
                        output_begin = input_lengths[batch_idx]
                        output_end = sequence_lengths[batch_idx][beam]
                        outputs = output_ids[batch_idx][beam][
                                  output_begin:output_end].tolist()
                        output_text = self.tokenizer.decode(outputs)
                        return {"output": output_text}
