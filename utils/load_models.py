import transformers
import os
import torch
from transformers import BitsAndBytesConfig

def split_model(model_name):
    device_map = {}
    if 'deepseek' in model_name:
        model_splits = {
            'deepseek-ai/deepseek-vl2-small': [13, 14], # 2 GPU for 16b
            'deepseek-ai/deepseek-vl2': [10, 10, 10], # 3 GPU for 27b
        }
        num_layers_per_gpu = model_splits[model_name]
        num_layers =  sum(num_layers_per_gpu)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision'] = 0
        device_map['projector'] = 0
        device_map['image_newline'] = 0
        device_map['view_seperator'] = 0
        device_map['language.model.embed_tokens'] = 0
        device_map['language.model.norm'] = 0
        device_map['language.lm_head'] = 0
        device_map[f'language.model.layers.{num_layers - 1}'] = 0
    elif 'llava' in model_name:
        num_layers_per_gpu = [17, 21, 21, 21]
        num_layers =  sum(num_layers_per_gpu)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_tower'] = 0
        device_map['multi_modal_projector'] = 0
        device_map['image_newline'] = 0
        device_map['view_seperator'] = 0
        device_map['language_model.model.rotary_emb'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        # device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    elif 'intern' in model_name:
        num_gpus = 4
        num_trans_layers = 32
        per_gpu_layers = 38 / num_gpus

        device_map = {
            'vit': 0,
            'vision_proj': 0,
            'model.tok_embeddings': 0,
            'plora_glb_GN': 0,
            'plora_sub_GN': 0,
            'model.norm': num_gpus - 1,
            'output': num_gpus - 1,
        }

        used = 3
        gpu_target = 0
        for i in range(num_trans_layers):
            if used >= per_gpu_layers:
                gpu_target += 1
                used = 0
            assert gpu_target < num_gpus
            device_map[f'model.layers.{i}'] = gpu_target
            used += 1
        
    return device_map


def load_i2t_model(engine, args=None):
    
    if 'otter' in engine:
        from Otter.src.otter_ai.models.otter.modeling_otter import OtterForConditionalGeneration
        model = OtterForConditionalGeneration.from_pretrained("luodian/OTTER-Image-LLaMA7B-LA-InContext", device_map="cuda", torch_dtype=torch.bfloat16)
        tokenizer = model.text_tokenizer
        image_processor = transformers.CLIPImageProcessor()
        processor = image_processor
    elif engine == 'deepseek-vl2-large':
        from transformers import AutoModelForCausalLM
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
        from deepseek_vl2.utils.io import load_pil_images
        from accelerate import dispatch_model, infer_auto_device_map
        from accelerate.utils import get_balanced_memory
        # specify the path to the model
        model_path = "deepseek-ai/deepseek-vl2"
        processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
        tokenizer = processor.tokenizer
        
        device_map = split_model(model_path)
        vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            device_map=device_map).eval()
        model = vl_gpt#.to(torch.bfloat16).eval()
    elif engine == 'deepseek-vl2':
        from transformers import AutoModelForCausalLM
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
        from deepseek_vl2.utils.io import load_pil_images
        # specify the path to the model
        model_path = "deepseek-ai/deepseek-vl2-small"
        processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
        tokenizer = processor.tokenizer
        vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True)
        model = vl_gpt.to(torch.bfloat16).cuda().eval()
    elif engine == 'llava16-7b':
        from llava.model.builder import load_pretrained_model as load_llava_model
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokenizer, model, image_processor, context_len = load_llava_model(model_path='lmms-lab/llava-next-interleave-qwen-7b', model_base=None,
                                                                            model_name='llava')

        processor = image_processor
        # tokenizer, model, image_processor, context_len = load_llava_model(model_path='liuhaotian/llava-v1.6-vicuna-7b', model_base=None, model_name='llava',
        #                                                                   device_map="cuda", torch_dtype=torch.bfloat16)
        # processor = processor.image_processor
    elif 'llava-onevision-0.5b' in engine:
        from llava.model.builder import load_pretrained_model as load_llava_model
        tokenizer, model, image_processor, context_len = load_llava_model(model_path='lmms-lab/llava-onevision-qwen2-0.5b-ov', model_base=None, attn_implementation="flash_attention_2",
                                                                          model_name='llava_qwen', device_map="cuda", torch_dtype=torch.bfloat16)
        processor = image_processor
    elif 'llava-onevision-7b-chat' in engine:
        from llava.model.builder import load_pretrained_model as load_llava_model
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokenizer, model, image_processor, context_len = load_llava_model(model_path='lmms-lab/llava-onevision-qwen2-7b-ov-chat', model_base=None,
                                                                            model_name='llava_qwen')

        processor = image_processor
    elif 'llava-onevision-7b' in engine:
        print(f'Loading model from {args.n_shot}, Run: {args.run}')
        from llava.model.builder import load_pretrained_model as load_llava_model
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokenizer, model, image_processor, context_len = load_llava_model(model_path='lmms-lab/llava-onevision-qwen2-7b-ov', model_base=None,
                                                                            model_name='llava_qwen')
        print(f'Finished loading model from {args.n_shot}, Run: {args.run}')
        
        processor = image_processor
    # elif 'llava-onevision-7b-ft' in engine:
    #     from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
    #     from peft import LoraConfig, get_peft_model
        
    #     model_path = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    #     org_processor = AutoProcessor.from_pretrained(model_path)
    #     processor = AutoProcessor.from_pretrained(model_path, chat_template=org_processor.chat_template)
    #     del org_processor
    #     image_processor = processor.image_processor
    #     tokenizer = processor.tokenizer

    #     # tokenizer.padding_side = 'left' # for batched generation with Qwen2
    #     model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, attn_implementation='flash_attention_2')

    #     target_modules=r'.*language_model.*\.(up_proj|k_proj|linear_2|down_proj|v_proj|q_proj|o_proj|gate_proj|linear_1)'


    #     config = LoraConfig(
    #         r=4, 
    #         lora_alpha=8, 
    #         target_modules=target_modules, 
    #         lora_dropout=0.05,
    #         bias="none", 
    #         task_type="CAUSAL_LM"
    #     )
    #     model = get_peft_model(model, config)
    #     model.cuda()
            
    #     lora_path = './fine_tune/finetuned_models/finetune_llava-onevision-7b_5_epochs_lr0.0001_42/step_14780'
    #     model.load_state_dict(torch.load(f'{lora_path}/checkpoint.pt'), strict=False)
    #     model.merge_and_unload()
        
    elif 'llava-onevision-72b' in engine:
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_path = "llava-hf/llava-onevision-qwen2-72b-ov-hf"
        
        device_map = split_model(model_path)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                # quantization_config=bnb_config,
                torch_dtype=torch.float16,
                # device_map=device_map
                )
        
        # model = AutoModelForCausalLM.from_pretrained(
        #         model_path, 
        #         trust_remote_code=True, 
        #         quantization_config=bnb_config,
        #         torch_dtype=torch.float16,
        #         device_map=device_map).eval()
        # model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        #         model_path, 
        #         trust_remote_code=True, 
        #         load_in_8bit=True,
        #         torch_dtype=torch.float16,
        #         device_map=device_map).eval()
        # for n, p in model.named_modules():
        #     print(n)
        # exit(0)
        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer
    elif engine == 'qwen-vl-2.5-instruct':
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
        # from qwen_vl_utils import process_vision_info
        
        # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2.5-VL-32B-Instruct",
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="auto",
        # )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-32B-Instruct", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")

        # from transformers.generation import GenerationConfig
        # tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct", trust_remote_code=True)
        # model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct", device_map="cuda", 
        #                                                           trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
        # for n, p in model.named_modules():
        #     print(n)
        # exit(0)
        # model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct", trust_remote_code=True)
        # processor = None
    elif engine == 'qwen-vl-instruct':
        from transformers.generation import GenerationConfig
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", 
                                                                  trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
        model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        processor = None
    elif engine == 'qwen-vl-chat':
        from transformers.generation import GenerationConfig
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", 
                                                                  trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
        model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        processor = None
    elif engine == 'qwen-vl':
        from transformers.generation import GenerationConfig
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
        processor = None
    elif engine == 'internlm-x2':
        # model = transformers.AutoModel.from_pretrained('internlm/internlm-xcomposer2-7b', trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda")
        # tokenizer = transformers.AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-7b', trust_remote_code=True)
        # model.tokenizer = tokenizer
        torch.set_grad_enabled(False)
        model = transformers.AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True, low_cpu_mem_usage=True).eval()
        model.half().cuda()
        from accelerate import dispatch_model
        from .utils import auto_configure_device_map
        device_map = split_model(engine)
        model = dispatch_model(model, device_map=device_map)
        tokenizer = transformers.AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
        model.tokenizer = tokenizer
        processor = None
    elif engine == 'openflamingo':
        from open_flamingo import create_model_and_transforms
        model, processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-7b",
            tokenizer_path="anas-awadalla/mpt-7b",
            cross_attn_every_n_layers=4,
        )
        model = model.to(torch.bfloat16).cuda()
    elif engine == 'idefics-9b-instruct':
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        checkpoint = "HuggingFaceM4/idefics-9b-instruct"
        model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
    elif engine == 'idefics-9b':
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        checkpoint = "HuggingFaceM4/idefics-9b"
        model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.float16, device_map="cuda", low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
    elif engine == 'idefics2-8b':
        from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
        processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b")
        model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b", torch_dtype=torch.float16, device_map="cuda", low_cpu_mem_usage=True)
        tokenizer = processor.tokenizer
    elif engine == 'idefics-80b-instruct':
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
        checkpoint = "HuggingFaceM4/idefics-80b-instruct"
        model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
    elif engine == 'gpt4v':
        model, tokenizer, processor = None, None, None
    elif 'gemini' in engine:
        model, tokenizer, processor = None, None, None
    else:
        raise NotImplementedError
        
        
    # elif engine == 'otter-mpt':
    #     from otter_ai import OtterForConditionalGeneration
    #     model = OtterForConditionalGeneration.from_pretrained("luodian/OTTER-Image-MPT7B", device_map="cuda", torch_dtype=torch.bfloat16)
    #     tokenizer = model.text_tokenizer
    #     image_processor = transformers.CLIPImageProcessor()
    #     processor = image_processor
    # elif engine == 'otter-llama':
    #     from otter_ai import OtterForConditionalGeneration
    #     model = OtterForConditionalGeneration.from_pretrained("luodian/OTTER-Image-LLaMA7B-LA-InContext", device_map="cuda", torch_dtype=torch.bfloat16)
    #     tokenizer = model.text_tokenizer
    #     image_processor = transformers.CLIPImageProcessor()
    #     processor = image_processor
    # elif engine == 'emu2-chat':
    #     from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
    #     tokenizer = transformers.AutoTokenizer.from_pretrained("BAAI/Emu2-Chat")
    #     with init_empty_weights():
    #         model = transformers.AutoModelForCausalLM.from_pretrained(
    #             "BAAI/Emu2-Chat",
    #             low_cpu_mem_usage=True,
    #             torch_dtype=torch.bfloat16,
    #             trust_remote_code=True).eval()
    #     # adjust according to your device
    #     device_map = infer_auto_device_map(model, max_memory={0:'38GiB',1:'38GiB',2:'38GiB',3:'38GiB'}, no_split_module_classes=['Block','LlamaDecoderLayer'])
    #     device_map["model.decoder.lm.lm_head"] = 0

    #     model = load_checkpoint_and_dispatch(
    #         model, 
    #         'path/to/models--BAAI--Emu2-Chat/snapshots/your_snapshot_path',
    #         device_map=device_map).eval()
    #     processor = None
    # elif engine == 'gpt4v':
    #     model, tokenizer, processor = None, None, None
    # else:
    #     raise NotImplementedError
    
    return model, tokenizer, processor

def load_t2i_model(engine, args):
    if engine == 'emu2-gen':
        # git clone https://github.com/baaivision/Emu/tree/main
        from Emu.Emu2.emu.diffusion import EmuVisualGeneration
        # git clone https://huggingface.co/BAAI/Emu2-Gen
        # set path to this folder
        path = "path/to/Emu2-Gen"
        tokenizer = transformers.AutoTokenizer.from_pretrained(f"{path}/tokenizer")

        # download model weigths from https://model.baai.ac.cn/model-detail/220122 
        pipe = EmuVisualGeneration.from_pretrained(
            'path/to/Emu2-Gen_pytorch_model.bf16.safetensors',
            dtype=torch.bfloat16,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        model = pipe
        processor = None
        model.multito(["cuda:0", "cuda:1",])
    elif engine == 'emu1-gen':
        # git clone https://github.com/baaivision/Emu/tree/main
        # git clone https://huggingface.co/BAAI/Emu
        from Emu.Emu1.models.modeling_emu import Emu
        from Emu.Emu1.models.pipeline import EmuGenerationPipeline
        import sys
        sys.path.append('path/to/Emu/Emu1')
        args = type('Args', (), {
            "instruct": False,
            "ckpt_path": 'path/to/Emu/pretrain', # huggingface weights
            "device": torch.device('cuda'),
        })()

        model = EmuGenerationPipeline.from_pretrained(
            path=args.ckpt_path,
            args=args,
        )
        tokenizer, processor = None, None
    elif engine == 'gill':
        from gill.gill.models import load_gill
        import sys
        # git clone https://github.com/kohjingyu/gill
        sys.path.append('path/to/gill')
        model_dir = 'path/to/gill/checkpoints/gill_opt'
        model = load_gill(model_dir, load_ret_embs=False)
        model = model.cuda()
        tokenizer, processor = None, None
    elif 'seed-llama' in engine:
        from omegaconf import OmegaConf
        import hydra, sys
        # git clone https://github.com/AILab-CVC/SEED
        os.environ['PROJECT_ROOT'] = 'path/to/SEED/'
        sys.path.append('path/to/SEED')
        from models.model_tools import get_pretrained_llama_causal_model
        from models import seed_llama_tokenizer
        tokenizer_cfg_path = f'path/to/SEED/configs/tokenizer/seed_llama_tokenizer_hf.yaml'
        tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
        tokenizer = hydra.utils.instantiate(
            tokenizer_cfg, device='cuda', load_diffusion=True)

        transform_cfg_path = f'path/to/SEED/configs/transform/clip_transform.yaml'
        transform_cfg = OmegaConf.load(transform_cfg_path)
        transform = hydra.utils.instantiate(transform_cfg)

        model_size = engine.split('-')[-1]
        model_cfg = OmegaConf.load(f'path/to/SEED/configs/llm/seed_llama_{model_size}.yaml')
        model = hydra.utils.instantiate(model_cfg, torch_dtype=torch.bfloat16)
        model = model.eval().cuda()
        processor = transform

    return model, tokenizer, processor