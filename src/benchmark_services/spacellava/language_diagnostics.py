"""Opt-in generation traces; never enabled for ordinary benchmark requests."""

import json
from pathlib import Path

from .diagnostics import compare_arrays
from ..common.files import atomic_json
from ..common.local_runtime import offline_imports


def token_trace(sequence_ids, generated_steps, max_new_tokens, eos_token_id):
    """Native inputs_embeds generation may prepend a dummy BOS, not the prompt."""
    if generated_steps < 0 or generated_steps > len(sequence_ids):
        raise ValueError("Generated-score count exceeds returned token count")
    prefix_length = len(sequence_ids) - generated_steps
    generated = sequence_ids[prefix_length:]
    eos_ids = eos_token_id if isinstance(eos_token_id, (tuple, list)) else [eos_token_id]
    ended_eos = bool(generated and generated[-1] in eos_ids)
    return {"returned_prefix_ids": sequence_ids[:prefix_length], "generated_token_ids": generated,
        "generated_steps": generated_steps, "ended_with_eos": ended_eos,
        "hit_token_budget": generated_steps >= max_new_tokens and not ended_eos}


def initial_past_state(kwargs):
    if "past_key_values" not in kwargs:
        return "not_passed"
    return "none" if kwargs["past_key_values"] is None else "provided"


def generation_arguments(request, settings):
    """Explicit diagnostic-only repeat controls; never enable sampling here."""
    arguments = {"do_sample": False, "num_beams": 1, "max_new_tokens": request.max_new_tokens,
        "use_cache": settings.get("use_cache", True), "return_dict_in_generate": True, "output_scores": True}
    if "repetition_penalty" in settings:
        value = settings["repetition_penalty"]
        import math
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid diagnostic repetition_penalty")
        arguments["repetition_penalty"] = value
    if "no_repeat_ngram_size" in settings:
        value = settings["no_repeat_ngram_size"]
        if type(value) is not int or value < 0:
            raise ValueError("Invalid diagnostic no_repeat_ngram_size")
        arguments["no_repeat_ngram_size"] = value
    return arguments


def generate_with_trace(adapter, request):
    options = adapter.options
    settings = options.get("diagnostic_request_settings", {}).get(request.request_id, {})
    template, cache = settings.get("conversation_template", "llava_v1"), settings.get("use_cache", True)
    folder = Path(options["diagnostic_output_dir"]).resolve()
    if "data_rover" in {p.casefold() for p in folder.parts}:
        raise ValueError("Protected archive cannot contain diagnostic outputs")
    path = folder / "generation_diagnostics.json"
    traces = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if request.request_id in traces:
        raise ValueError("Duplicate diagnostic request; use a new pilot run ID")
    cache_trace = {"requested": cache, "observed_forward_calls": 0,
                   "effective_modes": [], "unobserved_cache_calls": 0, "request_overridden": None,
                   "initial_past_key_values": "not_observed"}
    report = {"request_id": request.request_id, "cache_trace": cache_trace, "max_new_tokens": request.max_new_tokens}
    try:
        with offline_imports(Path(options["runtime_repo"]).resolve()), adapter.torch.inference_mode():
            ids, pixels, size, prepared = adapter.prepare_inputs(request, conversation_template=template, return_metadata=True)
            report.update(prepared)
            if report["image_token_count"] != 1:
                raise ValueError("Generation requires exactly one image token")
            report["generation_config"] = adapter.model.generation_config.to_dict()
            arguments = generation_arguments(request, settings)
            report["generation_arguments"] = arguments
            # The probe compares a direct forward with the first generation step.
            # Logits processors can legitimately alter scores: record, do not
            # automatically declare their difference to be a generation bug.
            direct = None
            if settings.get("compare_first_forward") is True:
                output = adapter.model(input_ids=ids, images=pixels, image_sizes=[size],
                                       use_cache=False, return_dict=True)
                direct = output.logits[0, -1, :].detach().float().cpu().numpy()
                del output
            # Transformers 4.37.2 forces caching for decoder-only inputs_embeds.
            # Observe the actual forward kwargs, not just our requested setting.
            # Register only after the direct probe; never change generation API.
            def observe_cache(module, args, kwargs):
                if cache_trace["observed_forward_calls"] == 0:
                    cache_trace["initial_past_key_values"] = initial_past_state(kwargs)
                cache_trace["observed_forward_calls"] += 1
                effective = kwargs.get("use_cache")
                if type(effective) is bool:
                    if effective not in cache_trace["effective_modes"]:
                        cache_trace["effective_modes"].append(effective)
                    if effective != cache:
                        cache_trace["request_overridden"] = True
                    elif cache_trace["request_overridden"] is None:
                        cache_trace["request_overridden"] = False
                else:
                    cache_trace["unobserved_cache_calls"] += 1
            handle = adapter.model.register_forward_pre_hook(observe_cache, with_kwargs=True)
            try:
                generated = adapter.model.generate(ids, images=pixels, image_sizes=[size], **arguments)
            finally:
                handle.remove()
            returned_ids = generated.sequences[0].detach().cpu().tolist()
            report.update(token_trace(returned_ids, len(generated.scores), request.max_new_tokens,
                                      adapter.model.generation_config.eos_token_id))
            report["returned_ids"] = returned_ids
            answer = adapter.tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()
            report["raw_response"] = answer
            report["decoded_with_special_tokens"] = adapter.tokenizer.batch_decode(generated.sequences, skip_special_tokens=False)[0]
            if generated.scores:
                first = generated.scores[0][0].detach().float().cpu().numpy()
                import numpy as np
                top = np.argsort(first)[-5:][::-1]
                report["first_token_top5"] = [{"token_id": int(i), "token": adapter.tokenizer.decode([int(i)]),
                    "score": float(first[i]) if np.isfinite(first[i]) else None} for i in top]
                if direct is not None:
                    report["direct_vs_generation_first_scores"] = compare_arrays(direct, first)
                    report["same_first_argmax"] = int(np.argmax(direct)) == int(np.argmax(first))
            report["success"] = True
    except Exception as error:
        report.update(success=False, error=f"{type(error).__name__}: {error}")
        traces[request.request_id] = report
        atomic_json(path, traces)
        raise
    traces[request.request_id] = report
    atomic_json(path, traces)
    return answer
