import torch
from torch.nn.functional import cross_entropy
from transformers import AutoModelForCausalLM, AutoTokenizer

#############################
##### Forward functions #####
#############################


def _forward(model, tokenizer, inputs, generate, no_grad, max_new_tokens=20, **kwargs):
    """
    if isinstance(inputs, dict) and 'input_ids' in inputs:
        tokenized = {k: v.to(model.device) for k, v in inputs.items()}
    elif "prepare_inputs" in kwargs:
        tokenized = kwargs["prepare_inputs"](model, tokenizer, inputs)
    else:
        tokenized = tokenizer(inputs, padding=True, return_tensors="pt").to(
            model.device
        )
    """
    device = torch.device("cuda")
    model.to(device)  
    tokenized = {k: v.to(model.device) for k, v in inputs.items()}
    
    if no_grad:
        with torch.no_grad():
            if generate:
                out = model.generate(
                    **tokenized,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    num_beams=1,
                )
            else:
                out = model.forward(**tokenized)
    else:
        if generate:
            out = model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,
            )
        else:
            out = model.forward(**tokenized)
    return out


def _generate_single(model, tokenizer, tokenized, no_grad):
    if no_grad:
        with torch.no_grad():
            out = model.generate(
                **tokenized,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_logits=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    else:
        out = model.generate(
            **tokenized,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_logits=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return out


######################################
##### Causal influence functions #####
######################################


def attn_head_influence(
    model,
    tokenizer,
    inputs,
    outputs,
    hooked_module,
    attn_head_activation=None,
    attn_head_idx=None,
):
    hook_handle = None
    input_ids, attention_mask, labels, label_mask = tokenize_for_ppl(
        model, tokenizer, inputs, outputs
    )

    def forward_pre_hook(module, input):
        if isinstance(input, tuple):
            new_input = input[0]
        else:
            new_input = input
        bsz, seq_len, _ = new_input.shape
        input_by_head = new_input.reshape(
            bsz, seq_len, model.config.num_attention_heads, -1
        )
        assert input_by_head.shape[-1] == attn_head_activation.shape[-1]
        prompt_idx = -torch.sum(label_mask, dim=1) - 1
        input_by_head[:, prompt_idx, attn_head_idx, :] = attn_head_activation.expand(
            bsz, -1
        )
        if isinstance(input, tuple):
            return (input_by_head.reshape(bsz, seq_len, -1),) + input[1:]
        return input_by_head.reshape(bsz, seq_len, -1)

    if attn_head_idx is not None and attn_head_activation is not None:
        hook_handle = hooked_module.register_forward_pre_hook(forward_pre_hook)
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits
        perplexity_batch = torch.exp(
            (
                cross_entropy(logits.transpose(1, 2), labels, reduction="none")
                * label_mask
            ).sum(1)
            / label_mask.sum(1)
        )
    if hook_handle is not None:
        hook_handle.remove()
    return sum(perplexity_batch.tolist())


##############################
##### Generate functions #####
##############################



def generate_add_layer_n(
    model,
    tokenizer,
    inputs,
    hooked_module,
    layer_activation,
    token_idx=-1,
    no_grad=True,
    max_new_tokens=10,
):
    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            new_output = output[0]
        else:
            new_output = output
        assert new_output.shape[-1] == layer_activation.shape[-1]
        new_output[:, token_idx, :] += layer_activation.expand(new_output.shape[0], -1)
        if isinstance(output, tuple):
            return (new_output,) + output[1:]
        return new_output

    return_dict = {
        "clean_logits": [],
        "corrupted_logits": [],
        "corrupted_sequences": [],
    }
    tokenized = tokenizer(inputs, padding=True, return_tensors="pt").to(model.device)
    for _ in range(max_new_tokens):
        hook_handle = hooked_module.register_forward_hook(forward_hook)
        corrupted_out = _generate_single(model, tokenizer, tokenized, no_grad=no_grad)
        return_dict["corrupted_logits"].append(corrupted_out.logits[0])
        return_dict["corrupted_sequences"].append(corrupted_out.sequences[:, -1])
        hook_handle.remove()
        clean_out = _generate_single(model, tokenizer, tokenized, no_grad=no_grad)
        return_dict["clean_logits"].append(clean_out.logits[0])
        attn_mask = tokenized["attention_mask"]
        attn_mask = torch.cat(
            [attn_mask, attn_mask.new_ones((attn_mask.shape[0], 1))], dim=-1
        )
        tokenized = {
            "input_ids": clean_out.sequences,
            "attention_mask": attn_mask,
        }

    return_dict["corrupted_sequences"] = torch.stack(
        return_dict["corrupted_sequences"], dim=1
    )
    return_dict["clean_sequences"] = clean_out.sequences
    return return_dict


def generate_add_layer_single(
    model,
    tokenizer,
    inputs,
    hooked_module,
    layer_activation,
    token_idx=-1,
    no_grad=True,
    max_new_tokens=10,
):
    hook_triggered = [False]

    def forward_hook(module, input, output):
        if hook_triggered[0]:
            return None
        if isinstance(output, tuple):
            new_output = output[0]
        else:
            new_output = output
        assert new_output.shape[-1] == layer_activation.shape[-1]
        new_output[:, token_idx, :] += layer_activation.expand(new_output.shape[0], -1)
        hook_triggered[0] = True
        if isinstance(output, tuple):
            return (new_output,) + output[1:]
        return new_output

    hook_handle = hooked_module.register_forward_hook(forward_hook)
    out = _forward(
        model,
        tokenizer,
        inputs,
        generate=True,
        no_grad=no_grad,
        max_new_tokens=max_new_tokens,
    )
    hook_handle.remove()

    return out


def generate_substitute_layer_single(
    model,
    tokenizer,
    inputs,
    hooked_modules,
    module_activations,
    sub_input_output,
    token_idx=0,
    no_grad=True,
    max_new_tokens=20,
    **kwargs,
):
    assert len(hooked_modules) == len(module_activations)
    if isinstance(token_idx, int):
        token_idx = [token_idx]
    hook_triggered = [False for _ in hooked_modules]

    def forward_pre_hook_idx(idx):
        def forward_pre_hook(module, input):
            if hook_triggered[idx]:
                return None
            if isinstance(input, tuple):
                new_input = input[0]
            else:
                new_input = input

            if "substitute_by_mask" in kwargs:
                for i in range(len(new_input)):
                    mask = kwargs["substitute_by_mask"][i]
                    new_input[i] = torch.cat(
                        [module_activations[idx][i, :mask, :], new_input[i][mask:, :]],
                        dim=0,
                    )
            else:
                new_activations = module_activations[idx].expand(-1, len(token_idx), -1)
                assert new_input[:, token_idx, :].shape == new_activations.shape
                new_input[:, token_idx, :] = new_activations

            hook_triggered[idx] = True
            if isinstance(input, tuple):
                return (new_input,) + input[1:]
            return new_input

        def forward_hook(module, input, output):
            if hook_triggered[idx]:
                return None
            if isinstance(output, tuple):
                new_output = output[0]
            else:
                new_output = output

            if "substitute_by_mask" in kwargs:
                for i in range(len(new_output)):
                    mask = kwargs["substitute_by_mask"][i]
                    new_output[i] = torch.cat(
                        [module_activations[idx][i, :mask, :], new_output[i][mask:, :]],
                        dim=0,
                    )
            else:
                new_activations = module_activations[idx].expand(-1, len(token_idx), -1)
                assert new_output[:, token_idx, :].shape == new_activations.shape
                new_output[:, token_idx, :] = new_activations

            hook_triggered[idx] = True
            if isinstance(output, tuple):
                return (new_output,) + output[1:]
            return new_output

        return forward_pre_hook if sub_input_output == "input" else forward_hook

    if sub_input_output == "input":
        hook_handles = [
            hooked_modules[i].register_forward_pre_hook(forward_pre_hook_idx(i))
            for i in range(len(hooked_modules))
        ]
    elif sub_input_output == "output":
        hook_handles = [
            hooked_modules[i].register_forward_hook(forward_pre_hook_idx(i))
            for i in range(len(hooked_modules))
        ]
    else:
        assert ValueError
    if "get_loss" in kwargs:
        assert "labels" in inputs
        out = _forward(
            model,
            tokenizer,
            inputs,
            generate=False,
            no_grad=(not kwargs["get_loss"]),
            **kwargs,
        )
    else:
        out = _forward(
            model,
            tokenizer,
            inputs,
            generate=True,
            no_grad=no_grad,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
    for hook_handle in hook_handles:
        hook_handle.remove()

    return out


def generate_add_attn_single(
    model,
    tokenizer,
    inputs,
    hooked_module,
    attn_head_idx,
    attn_head_activation,
    token_idx=-1,
    no_grad=True,
):
    hook_triggered = [False]

    def forward_pre_hook(module, input):
        if hook_triggered[0]:
            return None
        if isinstance(input, tuple):
            new_input = input[0]
        else:
            new_input = input
        bsz, seq_len, _ = new_input.shape
        input_by_head = new_input.reshape(
            bsz, seq_len, model.config.num_attention_heads, -1
        )
        input_by_head[:, token_idx, attn_head_idx, :] += attn_head_activation.expand(
            bsz, -1
        )
        hook_triggered[0] = True
        if isinstance(input, tuple):
            return (input_by_head.reshape(bsz, seq_len, -1),) + input[1:]
        return input_by_head.reshape(bsz, seq_len, -1)

    hook_handle = hooked_module.register_forward_pre_hook(forward_pre_hook)
    out = _forward(model, tokenizer, inputs, generate=True, no_grad=no_grad)
    hook_handle.remove()

    return out


#############################
##### Caching functions #####
#############################


def _forward_cache_outputs(
    model, tokenizer, inputs, hooked_modules, token_idx, no_grad=True, **kwargs
):
    cache = []

    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        if token_idx is None:
            cache.append(output)
        else:
            cache.append(output[:, token_idx, :])
        return None

    hook_handles = [
        hooked_module.register_forward_hook(forward_hook)
        for hooked_module in hooked_modules
    ]
    _ = _forward(model, tokenizer, inputs, generate=False, no_grad=no_grad, **kwargs)
    for hook_handle in hook_handles:
        hook_handle.remove()
    return cache


def _forward_cache_inputs(
    model, tokenizer, inputs, hooked_modules, split, token_idx, no_grad=True, **kwargs
):
    cache = []

    def forward_pre_hook_idx(idx):
        def forward_pre_hook(module, input):
            if isinstance(input, tuple):
                input = input[0]
            if split[idx]:
                bsz, seq_len, _ = input.shape
                input_by_head = input.reshape(
                    bsz, seq_len, model.config.num_attention_heads, -1
                )
                if token_idx is None:
                    cache.append(input_by_head)
                else:
                    cache.append(input_by_head[:, token_idx, :, :])
            else:
                if token_idx is None:
                    cache.append(input)
                else:
                    cache.append(input[:, token_idx, :])
            return None

        return forward_pre_hook

    hook_handles = [
        hooked_modules[i].register_forward_pre_hook(forward_pre_hook_idx(i))
        for i in range(len(hooked_modules))
    ]
    _ = _forward(model, tokenizer, inputs, generate=False, no_grad=no_grad, **kwargs)
    for hook_handle in hook_handles:
        hook_handle.remove()
    return cache


def cache_activations(
    model,
    tokenizer,
    module_list_or_str,
    cache_input_output,
    inputs,
    batch_size,
    token_idx=-1,
    split_attn_by_head=True,
    **kwargs,
):
    if isinstance(token_idx, int):
        token_idx = [token_idx]
    if isinstance(module_list_or_str, str):
        module_strs = [module_list_or_str]
    else:
        module_strs = module_list_or_str
    if split_attn_by_head and cache_input_output == "input":
        split = [True if "attn" in m else False for m in module_strs]
    else:
        split = [False for _ in module_strs]

    all_activations = [None for _ in module_strs]
    modules = []
    for m in module_strs:
        modules.append(eval(m))

    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]
        if cache_input_output == "input":
            activations = _forward_cache_inputs(
                model, tokenizer, batch, modules, split, token_idx, **kwargs
            )
        elif cache_input_output == "output":
            activations = _forward_cache_outputs(
                model, tokenizer, batch, modules, token_idx, **kwargs
            )
        else:
            raise ValueError("cache_input_output must be 'input' or 'output'")
        for j, activation in enumerate(activations):
            if i == 0:
                all_activations[j] = activation
            else:
                all_activations[j] = torch.cat([all_activations[j], activation], dim=0)
    # (num_modules, num_batches, [token_idx], activation_size)
    return all_activations

def cache_activations_multimodal(
    model,
    processor,  # Processor to handle both text and image inputs
    module_list_or_str,
    cache_input_output,
    inputs,  # this is implicitly already the batch size so just the total set tbh
    batch_size,
    token_idx,
    split_attn_by_head=True,
    **kwargs,
):
    if isinstance(token_idx, int):
        token_idx = [token_idx]
    if isinstance(module_list_or_str, str):
        module_strs = [module_list_or_str]
    else:
        module_strs = module_list_or_str
    if split_attn_by_head and cache_input_output == "input":
        split = [True if "attn" in m else False for m in module_strs]
    else:
        split = [False for _ in module_strs]

    #all_activations = [None for _ in module_strs]
    modules = []
    for m in module_strs:
        modules.append(eval(m)) 
    
    if cache_input_output == "input":
        activations = _forward_cache_inputs(
            model, processor.tokenizer, inputs, modules, split, token_idx, **kwargs
        )
    elif cache_input_output == "output":
        activations = _forward_cache_outputs(
            model, processor.tokenizer, inputs, modules, token_idx, **kwargs
        )
    else:
        raise ValueError("cache_input_output must be 'input' or 'output'")
    """
    for j, activation in enumerate(activations):
        all_activations[j] = activation
    """
    return activations

def batched_cache_activations_multimodal(
    model,
    processor,  
    module_list_or_str,
    cache_input_output,
    inputs,  # full
    batch_size,
    token_idx,
    split_attn_by_head=True,
    **kwargs,
):
    if isinstance(token_idx, int):
        token_idx = [token_idx]
    if isinstance(module_list_or_str, str):
        module_strs = [module_list_or_str]
    else:
        module_strs = module_list_or_str
    if split_attn_by_head and cache_input_output == "input":
        split = [True if "attn" in m else False for m in module_strs]
    else:
        split = [False for _ in module_strs]

    modules = []
    for m in module_strs:
        modules.append(eval(m)) 

    def process_batch(batch):
        if cache_input_output == "input":
            return _forward_cache_inputs(
                model, processor.tokenizer, batch, modules, split, token_idx, **kwargs
            )
        elif cache_input_output == "output":
            return _forward_cache_outputs(
                model, processor.tokenizer, batch, modules, token_idx, **kwargs
            )
        else:
            raise ValueError("cache_input_output must be 'input' or 'output'")

    num_batches = len(inputs['input_ids']) // batch_size + (0 if len(inputs['input_ids']) % batch_size == 0 else 1)
    all_activations = [[] for _ in module_strs]
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(inputs['input_ids']))
        batch = {key: value[batch_start:batch_end] for key, value in inputs.items()}
        activations = process_batch(batch)
        for i in range(len(all_activations)):
            all_activations[i].extend(activations[i])  
        print(len(all_activations))
        print(len(all_activations[0]))
        print(f"Batch {i}")

    return all_activations


##############################
##### Model loading code #####
##############################


def load_model(args):
    model_name_or_path = args.model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    if any([n in model_name_or_path for n in ["llama", "zephyr", "gemma", "mistral", "Qwen", "llava"]]):
        module_str_dict = {
            "layer": "model.model.layers[{layer_idx}]",
            "attn": "model.model.layers[{layer_idx}].self_attn.o_proj",
        }
        n_layers = len(model.model.layers)
    elif "gpt-j" in model_name_or_path:
        module_str_dict = {
            "layer": "model.transformer.h[{layer_idx}]",
            "attn": "model.transformer.h[{layer_idx}].attn.o_proj",
        }
        n_layers = len(model.transformer.h)
    elif "opt" in model_name_or_path:
        module_str_dict = {
            "layer": "model.model.decoder.layers[{layer_idx}]",
            "attn": "model.model.decoder.layers[{layer_idx}].self_attn.o_proj",
        }
        n_layers = len(model.model.decoder.layers)
    args.module_str_dict = module_str_dict
    args.n_layers = n_layers
    return model, tokenizer


def get_modules(model):
    modules = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            modules.append(name)
    return modules


def print_hooks(module):
    no_hooks = False
    # Print forward hooks
    if hasattr(module, "_forward_hooks") and len(module._forward_hooks) > 0:
        no_hooks = True
        print(f"{module.__class__.__name__} has the following forward hooks:")
        for hook_id, hook in module._forward_hooks.items():
            print(f"\tHook ID: {hook_id}, Hook: {hook}")

    # Print backward hooks
    if hasattr(module, "_backward_hooks") and len(module._backward_hooks) > 0:
        no_hooks = True
        print(f"{module.__class__.__name__} has the following backward hooks:")
        for hook_id, hook in module._backward_hooks.items():
            print(f"\tHook ID: {hook_id}, Hook: {hook}")

    # Print forward pre-hooks
    if hasattr(module, "_forward_pre_hooks") and len(module._forward_pre_hooks) > 0:
        no_hooks = True
        print(f"{module.__class__.__name__} has the following forward pre-hooks:")
        for hook_id, hook in module._forward_pre_hooks.items():
            print(f"\tHook ID: {hook_id}, Hook: {hook}")

    if no_hooks:
        print(f"{module.__class__.__name__} has no hooks.")