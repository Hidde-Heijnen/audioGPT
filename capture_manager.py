
class CaptureManager:
    def __init__(self, model, points, listen_index=-1):
        self.model = model
        self.points = points
        self.listen_index = listen_index
        self.captures = {}
        self.hooks = []

    def _capture_hook(self, name):
        def hook(module, input, output):
            val = output.detach().cpu()
            if self.listen_index is not None:
                idx = self.listen_index if self.listen_index >= 0 else val.shape[1] + self.listen_index
                val = val[:, idx:idx+1, :]
            self.captures[name] = val
        return hook

    def _capture_head_hook(self, name, head_idx=None):
        def hook(module, input, output):
            val = output.detach().cpu()
            if head_idx is not None:
                val = val[:, head_idx:head_idx+1, :, :]
            if self.listen_index is not None:
                idx = self.listen_index if self.listen_index >= 0 else val.shape[2] + self.listen_index
                val = val[:, :, idx:idx+1, :]
            self.captures[name] = val
        return hook

    def _capture_attn_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]  # token path
            val = output.detach().cpu()
            if self.listen_index is not None:
                idx = self.listen_index if self.listen_index >= 0 else val.shape[1] + self.listen_index
                val = val[:, idx:idx+1, :]
            self.captures[name] = val
        return hook

    def __enter__(self):
        for point in self.points:
            if not isinstance(point, tuple):
                point = (point,)
            key = point[0]
            if key == 'wte':
                h = self.model.transformer.wte.register_forward_hook(self._capture_hook('wte'))
                self.hooks.append(h)
            elif key == 'w_audio' and hasattr(self.model.transformer, 'w_audio'):
                h = self.model.transformer.w_audio.register_forward_hook(self._capture_hook('w_audio'))
                self.hooks.append(h)
            elif key == 'after_posenc' and hasattr(self.model, 'pos_add') and self.model.pos_add is not None:
                h = self.model.pos_add.register_forward_hook(self._capture_hook('after_posenc'))
                self.hooks.append(h)
            elif key == 'final_ln':
                h = self.model.transformer.ln_f.register_forward_hook(self._capture_hook('final_ln'))
                self.hooks.append(h)
            elif key == 'audio_final_ln' and hasattr(self.model, 'audio_ln_f'):
                h = self.model.audio_ln_f.register_forward_hook(self._capture_hook('audio_final_ln'))
                self.hooks.append(h)
            elif isinstance(key, int):
                layer_idx = key
                block = self.model.transformer.h[layer_idx]
                subkey = point[1] if len(point) > 1 else None
                head_idx = point[2] if len(point) > 2 else None
                if subkey == 'after_ln1':
                    h = block.ln_1.register_forward_hook(self._capture_hook(f'layer{layer_idx}_after_ln1'))
                    self.hooks.append(h)
                elif subkey == 'after_audio_ln1' and hasattr(block, 'audio_ln1'):
                    h = block.audio_ln1.register_forward_hook(self._capture_hook(f'layer{layer_idx}_after_audio_ln1'))
                    self.hooks.append(h)
                elif subkey == 'after_attn':
                    if head_idx is not None:
                        h = block.attn.head_out.register_forward_hook(self._capture_head_hook(f'layer{layer_idx}_after_attn_head{head_idx}', head_idx))
                    else:
                        h = block.attn.register_forward_hook(self._capture_attn_hook(f'layer{layer_idx}_after_attn'))
                    self.hooks.append(h)
                elif subkey == 'after_audio_attn' and hasattr(block.attn, 'head_audio_out'):
                    if head_idx is not None:
                        h = block.attn.head_audio_out.register_forward_hook(self._capture_head_hook(f'layer{layer_idx}_after_audio_attn_head{head_idx}', head_idx))
                    else:
                        h = block.attn.audio_mix_mean.register_forward_hook(self._capture_hook(f'layer{layer_idx}_after_audio_attn'))
                    self.hooks.append(h)
                elif subkey == 'after_attn_resid':
                    h = block.attn_resid.register_forward_hook(self._capture_hook(f'layer{layer_idx}_after_attn_resid'))
                    self.hooks.append(h)
                elif subkey == 'after_audio_attn_resid' and hasattr(block, 'audio_attn_resid'):
                    h = block.audio_attn_resid.register_forward_hook(self._capture_hook(f'layer{layer_idx}_after_audio_attn_resid'))
                    self.hooks.append(h)
                elif subkey == 'after_ln2':
                    h = block.ln_2.register_forward_hook(self._capture_hook(f'layer{layer_idx}_after_ln2'))
                    self.hooks.append(h)
                elif subkey == 'after_mlp':
                    h = block.mlp.register_forward_hook(self._capture_hook(f'layer{layer_idx}_after_mlp'))
                    self.hooks.append(h)
                elif subkey == 'after_mlp_resid':
                    h = block.mlp_resid.register_forward_hook(self._capture_hook(f'layer{layer_idx}_after_mlp_resid'))
                    self.hooks.append(h)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []