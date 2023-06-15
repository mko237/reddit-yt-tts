from tortoise.api import TextToSpeech, MODELS_DIR
from tortoise.api import *
from tortoise.utils.audio import load_voices
from tortoise.utils.text import split_and_recombine_text

import torch, torchaudio
import os
import time


class TTSOPT(TextToSpeech):

    
    def __init__(self, autoregressive_batch_size=None, models_dir=MODELS_DIR, enable_redaction=True, device=None):
        """
        Constructor
        :param autoregressive_batch_size: Specifies how many samples to generate per batch. Lower this if you are seeing
                                          GPU OOM errors. Larger numbers generates slightly faster.
        :param models_dir: Where model weights are stored. This should only be specified if you are providing your own
                           models, otherwise use the defaults.
        :param enable_redaction: When true, text enclosed in brackets are automatically redacted from the spoken output
                                 (but are still rendered by the model). This can be used for prompt engineering.
                                 Default is true.
        :param device: Device to use when running the model. If omitted, the device will be automatically chosen.
        """
        self.models_dir = models_dir
        self.autoregressive_batch_size = pick_best_batch_size_for_gpu() if autoregressive_batch_size is None else autoregressive_batch_size
        self.enable_redaction = enable_redaction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.enable_redaction:
            self.aligner = Wav2VecAlignment()

        self.tokenizer = VoiceBpeTokenizer()

        if os.path.exists(f'{models_dir}/autoregressive.ptt'):
            # Assume this is a traced directory.
            self.autoregressive = torch.jit.load(f'{models_dir}/autoregressive.ptt')
            self.diffusion = torch.jit.load(f'{models_dir}/diffusion_decoder.ptt')
        else:
            self.autoregressive = UnifiedVoice(max_mel_tokens=604, max_text_tokens=402, max_conditioning_inputs=2, layers=30,
                                          model_dim=1024,
                                          heads=16, number_text_tokens=255, start_text_token=255, checkpointing=False,
                                          train_solo_embeddings=False).eval() # mkm removed .cpu().eval() 
            self.autoregressive.load_state_dict(torch.load(get_model_path('autoregressive.pth', models_dir)))

            self.diffusion = DiffusionTts(model_channels=1024, num_layers=10, in_channels=100, out_channels=200,
                                          in_latent_channels=1024, in_tokens=8193, dropout=0, use_fp16=False, num_heads=16,
                                          layer_drop=0, unconditioned_percentage=0).eval() #mkm removed .cpu().eval()
            self.diffusion.load_state_dict(torch.load(get_model_path('diffusion_decoder.pth', models_dir)))

        self.clvp = CLVP(dim_text=768, dim_speech=768, dim_latent=768, num_text_tokens=256, text_enc_depth=20,
                         text_seq_len=350, text_heads=12,
                         num_speech_tokens=8192, speech_enc_depth=20, speech_heads=12, speech_seq_len=430,
                         use_xformers=True).eval() # mkm removed .cpu().eval() 

        self.clvp.load_state_dict(torch.load(get_model_path('clvp2.pth', models_dir)))
        self.cvvp = None # CVVP model is only loaded if used.

        self.vocoder = UnivNetGenerator().cpu()
        self.vocoder.load_state_dict(torch.load(get_model_path('vocoder.pth', models_dir), map_location=torch.device('cpu'))['model_g'])
        self.vocoder.eval(inference=True)

        # mkm caching results
        self.gcl_cache = []

        # Random latent generators (RLGs) are loaded lazily.
        self.rlg_auto = None
        self.rlg_diffusion = None

    def get_conditioning_latents_to_cache(self, voice_samples, return_mels=False):
        """
        mkm: duplicating function to make lineprofiler easier to read. This function wont be read by profiler so that the actual function can be profiled only after caching. instead of once with and without caching
        """
        if len(self.gcl_cache) > 0:
            return self.gcl_cache

        with torch.no_grad():
            voice_samples = [v.to(self.device) for v in voice_samples]

            auto_conds = []
            if not isinstance(voice_samples, list):
                voice_samples = [voice_samples]
            for vs in voice_samples:
                auto_conds.append(format_conditioning(vs, device=self.device))
            auto_conds = torch.stack(auto_conds, dim=1)
            self.autoregressive = self.autoregressive.to(self.device)
            auto_latent = self.autoregressive.get_conditioning(auto_conds)
            #mkm comment self.autoregressive = self.autoregressive.cpu()

            diffusion_conds = []
            for sample in voice_samples:
                # The diffuser operates at a sample rate of 24000 (except for the latent inputs)
                sample = torchaudio.functional.resample(sample, 22050, 24000)
                sample = pad_or_truncate(sample, 102400)
                cond_mel = wav_to_univnet_mel(sample.to(self.device), do_normalization=False, device=self.device)
                diffusion_conds.append(cond_mel)
            diffusion_conds = torch.stack(diffusion_conds, dim=1)

            self.diffusion = self.diffusion.to(self.device)
            diffusion_latent = self.diffusion.get_conditioning(diffusion_conds)
            #mkm comment self.diffusion = self.diffusion.cpu()

        if return_mels:
            # mkm replaced: return auto_latent, diffusion_latent, auto_conds, diffusion_conds
            self.gcl_cache = auto_latent, diffusion_latent, auto_conds, diffusion_conds
            return self.gcl_cache
        else:
            return auto_latent, diffusion_latent

    
    def get_conditioning_latents(self, voice_samples, return_mels=False):
        """
        Transforms one or more voice_samples into a tuple (autoregressive_conditioning_latent, diffusion_conditioning_latent).
        These are expressive learned latents that encode aspects of the provided clips like voice, intonation, and acoustic
        properties.
        :param voice_samples: List of 2 or more ~10 second reference clips, which should be torch tensors containing 22.05kHz waveform data.
        """
        if len(self.gcl_cache) > 0:
            return self.gcl_cache

        with torch.no_grad():
            voice_samples = [v.to(self.device) for v in voice_samples]

            auto_conds = []
            if not isinstance(voice_samples, list):
                voice_samples = [voice_samples]
            for vs in voice_samples:
                auto_conds.append(format_conditioning(vs, device=self.device))
            auto_conds = torch.stack(auto_conds, dim=1)
            self.autoregressive = self.autoregressive.to(self.device)
            auto_latent = self.autoregressive.get_conditioning(auto_conds)
            #mkm comment self.autoregressive = self.autoregressive.cpu()

            diffusion_conds = []
            for sample in voice_samples:
                # The diffuser operates at a sample rate of 24000 (except for the latent inputs)
                sample = torchaudio.functional.resample(sample, 22050, 24000)
                sample = pad_or_truncate(sample, 102400)
                cond_mel = wav_to_univnet_mel(sample.to(self.device), do_normalization=False, device=self.device)
                diffusion_conds.append(cond_mel)
            diffusion_conds = torch.stack(diffusion_conds, dim=1)

            self.diffusion = self.diffusion.to(self.device)
            diffusion_latent = self.diffusion.get_conditioning(diffusion_conds)
            #mkm comment self.diffusion = self.diffusion.cpu()

        if return_mels:
            # mkm replaced: return auto_latent, diffusion_latent, auto_conds, diffusion_conds
            self.gcl_cache = auto_latent, diffusion_latent, auto_conds, diffusion_conds
            return self.gcl_cache
        else:
            return auto_latent, diffusion_latent

    
    def tts_with_preset(self, text, preset='fast', **kwargs):
        """
        Calls TTS with one of a set of preset generation parameters. Options:
            'ultra_fast': Produces speech at a speed which belies the name of this repo. (Not really, but it's definitely fastest).
            'fast': Decent quality speech at a decent inference rate. A good choice for mass inference.
            'standard': Very good quality. This is generally about as good as you are going to get.
            'high_quality': Use if you want the absolute best. This is not really worth the compute, though.
        """
        # Use generally found best tuning knobs for generation.
        settings = {'temperature': .8, 'length_penalty': 1.0, 'repetition_penalty': 2.0,
                    'top_p': .8,
                    'cond_free_k': 2.0, 'diffusion_temperature': 1.0}
        # Presets are defined here.
        presets = {
            'ultra_fast': {'num_autoregressive_samples': 16, 'diffusion_iterations': 30, 'cond_free': False},
            'fast': {'num_autoregressive_samples': 96, 'diffusion_iterations': 80},
            'standard': {'num_autoregressive_samples': 256, 'diffusion_iterations': 200},
            'high_quality': {'num_autoregressive_samples': 256, 'diffusion_iterations': 400},
        }
        settings.update(presets[preset])
        settings.update(kwargs) # allow overriding of preset settings with kwargs
        return self.tts(text, **settings)

    
    def tts(self, text, voice_samples=None, conditioning_latents=None, k=1, verbose=True, use_deterministic_seed=None,
            return_deterministic_state=False,
            # autoregressive generation parameters follow
            num_autoregressive_samples=512, temperature=.8, length_penalty=1, repetition_penalty=2.0, top_p=.8, max_mel_tokens=500,
            # CVVP parameters follow
            cvvp_amount=.0,
            # diffusion generation parameters follow
            diffusion_iterations=100, cond_free=True, cond_free_k=2, diffusion_temperature=1.0,
            **hf_generate_kwargs):
        """
        Produces an audio clip of the given text being spoken with the given reference voice.
        :param text: Text to be spoken.
        :param voice_samples: List of 2 or more ~10 second reference clips which should be torch tensors containing 22.05kHz waveform data.
        :param conditioning_latents: A tuple of (autoregressive_conditioning_latent, diffusion_conditioning_latent), which
                                     can be provided in lieu of voice_samples. This is ignored unless voice_samples=None.
                                     Conditioning latents can be retrieved via get_conditioning_latents().
        :param k: The number of returned clips. The most likely (as determined by Tortoises' CLVP model) clips are returned.
        :param verbose: Whether or not to print log messages indicating the progress of creating a clip. Default=true.
        ~~AUTOREGRESSIVE KNOBS~~
        :param num_autoregressive_samples: Number of samples taken from the autoregressive model, all of which are filtered using CLVP.
               As Tortoise is a probabilistic model, more samples means a higher probability of creating something "great".
        :param temperature: The softmax temperature of the autoregressive model.
        :param length_penalty: A length penalty applied to the autoregressive decoder. Higher settings causes the model to produce more terse outputs.
        :param repetition_penalty: A penalty that prevents the autoregressive decoder from repeating itself during decoding. Can be used to reduce the incidence
                                   of long silences or "uhhhhhhs", etc.
        :param top_p: P value used in nucleus sampling. (0,1]. Lower values mean the decoder produces more "likely" (aka boring) outputs.
        :param max_mel_tokens: Restricts the output length. (0,600] integer. Each unit is 1/20 of a second.
        :param typical_sampling: Turns typical sampling on or off. This sampling mode is discussed in this paper: https://arxiv.org/abs/2202.00666
                                 I was interested in the premise, but the results were not as good as I was hoping. This is off by default, but
                                 could use some tuning.
        :param typical_mass: The typical_mass parameter from the typical_sampling algorithm.
        ~~CLVP-CVVP KNOBS~~
        :param cvvp_amount: Controls the influence of the CVVP model in selecting the best output from the autoregressive model.
                            [0,1]. Values closer to 1 mean the CVVP model is more important, 0 disables the CVVP model.
        ~~DIFFUSION KNOBS~~
        :param diffusion_iterations: Number of diffusion steps to perform. [0,4000]. More steps means the network has more chances to iteratively refine
                                     the output, which should theoretically mean a higher quality output. Generally a value above 250 is not noticeably better,
                                     however.
        :param cond_free: Whether or not to perform conditioning-free diffusion. Conditioning-free diffusion performs two forward passes for
                          each diffusion step: one with the outputs of the autoregressive model and one with no conditioning priors. The output
                          of the two is blended according to the cond_free_k value below. Conditioning-free diffusion is the real deal, and
                          dramatically improves realism.
        :param cond_free_k: Knob that determines how to balance the conditioning free signal with the conditioning-present signal. [0,inf].
                            As cond_free_k increases, the output becomes dominated by the conditioning-free signal.
                            Formula is: output=cond_present_output*(cond_free_k+1)-cond_absenct_output*cond_free_k
        :param diffusion_temperature: Controls the variance of the noise fed into the diffusion model. [0,1]. Values at 0
                                      are the "mean" prediction of the diffusion network and will sound bland and smeared.
        ~~OTHER STUFF~~
        :param hf_generate_kwargs: The huggingface Transformers generate API is used for the autoregressive transformer.
                                   Extra keyword args fed to this function get forwarded directly to that API. Documentation
                                   here: https://huggingface.co/docs/transformers/internal/generation_utils
        :return: Generated audio clip(s) as a torch tensor. Shape 1,S if k=1 else, (k,1,S) where S is the sample length.
                 Sample rate is 24kHz.
        """
        deterministic_seed = self.deterministic_state(seed=use_deterministic_seed)

        text_tokens = torch.IntTensor(self.tokenizer.encode(text)).unsqueeze(0).to(self.device)
        text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
        assert text_tokens.shape[-1] < 400, 'Too much text provided. Break the text up into separate segments and re-try inference.'

        auto_conds = None
        if voice_samples is not None:
            auto_conditioning, diffusion_conditioning, auto_conds, _ = self.get_conditioning_latents(voice_samples, return_mels=True)
        elif conditioning_latents is not None:
            auto_conditioning, diffusion_conditioning = conditioning_latents
        else:
            auto_conditioning, diffusion_conditioning = self.get_random_conditioning_latents()
        auto_conditioning = auto_conditioning.to(self.device)
        diffusion_conditioning = diffusion_conditioning.to(self.device)

        diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=diffusion_iterations, cond_free=cond_free, cond_free_k=cond_free_k)

        with torch.no_grad():
            samples = []
            num_batches = num_autoregressive_samples // self.autoregressive_batch_size
            stop_mel_token = self.autoregressive.stop_mel_token
            calm_token = 83  # This is the token for coding silence, which is fixed in place with "fix_autoregressive_output"
            self.autoregressive = self.autoregressive.to(self.device)
            if verbose:
                print("Generating autoregressive samples..")
            for b in tqdm(range(num_batches), disable=not verbose):
                codes = self.autoregressive.inference_speech(auto_conditioning, text_tokens,
                                                             do_sample=True,
                                                             top_p=top_p,
                                                             temperature=temperature,
                                                             num_return_sequences=self.autoregressive_batch_size,
                                                             length_penalty=length_penalty,
                                                             repetition_penalty=repetition_penalty,
                                                             max_generate_length=max_mel_tokens,
                                                             **hf_generate_kwargs)
                padding_needed = max_mel_tokens - codes.shape[1]
                codes = F.pad(codes, (0, padding_needed), value=stop_mel_token)
                samples.append(codes)
            #mkm comment self.autoregressive = self.autoregressive.cpu()

            clip_results = []
            self.clvp = self.clvp.to(self.device)
            if cvvp_amount > 0:
                if self.cvvp is None:
                    self.load_cvvp()
                self.cvvp = self.cvvp.to(self.device)
            if verbose:
                if self.cvvp is None:
                    print("Computing best candidates using CLVP")
                else:
                    print(f"Computing best candidates using CLVP {((1-cvvp_amount) * 100):2.0f}% and CVVP {(cvvp_amount * 100):2.0f}%")
            for batch in tqdm(samples, disable=not verbose):
                for i in range(batch.shape[0]):
                    batch[i] = fix_autoregressive_output(batch[i], stop_mel_token)
                if cvvp_amount != 1:
                    clvp = self.clvp(text_tokens.repeat(batch.shape[0], 1), batch, return_loss=False)
                if auto_conds is not None and cvvp_amount > 0:
                    cvvp_accumulator = 0
                    for cl in range(auto_conds.shape[1]):
                        cvvp_accumulator = cvvp_accumulator + self.cvvp(auto_conds[:, cl].repeat(batch.shape[0], 1, 1), batch, return_loss=False)
                    cvvp = cvvp_accumulator / auto_conds.shape[1]
                    if cvvp_amount == 1:
                        clip_results.append(cvvp)
                    else:
                        clip_results.append(cvvp * cvvp_amount + clvp * (1-cvvp_amount))
                else:
                    clip_results.append(clvp)
            clip_results = torch.cat(clip_results, dim=0)
            samples = torch.cat(samples, dim=0)
            best_results = samples[torch.topk(clip_results, k=k).indices]
            #mkm comment self.clvp = self.clvp.cpu()
            if self.cvvp is not None:
                self.cvvp = self.cvvp.cpu()
            del samples

            # The diffusion model actually wants the last hidden layer from the autoregressive model as conditioning
            # inputs. Re-produce those for the top results. This could be made more efficient by storing all of these
            # results, but will increase memory usage.
            #mkm comment self.autoregressive = self.autoregressive.to(self.device)
            best_latents = self.autoregressive(auto_conditioning.repeat(k, 1), text_tokens.repeat(k, 1),
                                               torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), best_results,
                                               torch.tensor([best_results.shape[-1]*self.autoregressive.mel_length_compression], device=text_tokens.device),
                                               return_latent=True, clip_inputs=False)
            #mkm comment self.autoregressive = self.autoregressive.cpu()
            del auto_conditioning

            if verbose:
                print("Transforming autoregressive outputs into audio..")
            wav_candidates = []
            self.diffusion = self.diffusion.to(self.device)
            self.vocoder = self.vocoder.to(self.device)
            for b in range(best_results.shape[0]):
                codes = best_results[b].unsqueeze(0)
                latents = best_latents[b].unsqueeze(0)

                # Find the first occurrence of the "calm" token and trim the codes to that.
                ctokens = 0
                for k in range(codes.shape[-1]):
                    if codes[0, k] == calm_token:
                        ctokens += 1
                    else:
                        ctokens = 0
                    if ctokens > 8:  # 8 tokens gives the diffusion model some "breathing room" to terminate speech.
                        latents = latents[:, :k]
                        break

                mel = do_spectrogram_diffusion(self.diffusion, diffuser, latents, diffusion_conditioning,
                                               temperature=diffusion_temperature, verbose=verbose)
                wav = self.vocoder.inference(mel)
                wav_candidates.append(wav.cpu())
            #mkm comment self.diffusion = self.diffusion.cpu()
            self.vocoder = self.vocoder.cpu()

            def potentially_redact(clip, text):
                if self.enable_redaction:
                    return self.aligner.redact(clip.squeeze(1), text).unsqueeze(1)
                return clip
            wav_candidates = [potentially_redact(wav_candidate, text) for wav_candidate in wav_candidates]

            if len(wav_candidates) > 1:
                res = wav_candidates
            else:
                res = wav_candidates[0]

            if return_deterministic_state:
                return res, (deterministic_seed, text, voice_samples, conditioning_latents)
            else:
                return res


def exec_tts(tts_text,output_path,selected_voice="train_empire"):
    #tts = TextToSpeech(models_dir=MODELS_DIR)
    print(f"Current device: {tts.device}")

    candidates=1
    cvvp_amount=0
    seed=None
    preset= 'fast' #'standard' #'fast' #standard

    texts = split_and_recombine_text(tts_text)

    if '&' in selected_voice:
        voice_sel = selected_voice.split('&')
    else:
        voice_sel = [selected_voice]
    voice_samples, conditioning_latents = load_voices(voice_sel)

    all_parts = []
    for j, text in enumerate(texts):
        gen, dbg_state = tts.tts_with_preset(text, k=candidates, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                              preset=preset, use_deterministic_seed=seed, return_deterministic_state=True, cvvp_amount=cvvp_amount,verbose=False)
        gen = gen.squeeze(0).cpu()
        all_parts.append(gen)

    full_audio = torch.cat(all_parts, dim=-1)
    torchaudio.save(os.path.join(output_path), full_audio, 24000)

# print("initializing tts api")
# tts = TTSOPT(models_dir=MODELS_DIR)

# #caching function results to increase speed of tts jobs
# print("populating cache")
# voice_samples, conditioning_latents = load_voices(["train_empire"])
# tts.get_conditioning_latents_to_cache(voice_samples, return_mels=True)


# time_s = time.time()
# print("starting. first..")
# exec_tts("test audio","test2.wav")
# time_e = time.time()
# time_duration = time_e - time_s
# print(f"first completed took {time_duration} secs")

# time_s = time.time()
# print("starting second...")
# exec_tts("test audio","test3.wav")
# time_e = time.time()
# time_duration = time_e - time_s
# print(f"second completed took {time_duration} secs")
