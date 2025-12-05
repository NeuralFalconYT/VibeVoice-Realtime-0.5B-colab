# %%writefile /content/VibeVoice/app.py
import sys
import os
import gradio as gr
import torch
import time
import copy
from pathlib import Path
from typing import Optional, Tuple

# ==========================================
# 1. SETUP & PATHS
# ==========================================
vibevoice_path = f"{os.getcwd()}/vibevoice"
sys.path.append(vibevoice_path)

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)

# ==========================================
# 2. VOICE MAPPER
# ==========================================
class VoiceMapper:
    def __init__(self):
        self.voice_presets = {}
        self.available_voices = {}
        self.setup_voice_presets()

    def setup_voice_presets(self):
        voices_dir = "./demo/voices/streaming_model/"
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            return

        pt_files = [
            f for f in os.listdir(voices_dir)
            if f.lower().endswith(".pt") and os.path.isfile(os.path.join(voices_dir, f))
        ]

        for pt_file in pt_files:
            name = os.path.splitext(pt_file)[0]
            full_path = os.path.join(voices_dir, pt_file)
            self.available_voices[name] = full_path

        self.available_voices = dict(sorted(self.available_voices.items()))

    def get_voice_path(self, name: str):
        return self.available_voices.get(name)

# ==========================================
# 3. MODEL LOADING
# ==========================================
print("⏳ Loading Model...")
if os.path.exists("/content/models/VibeVoice-Realtime-0.5B/model.safetensors"):
  MODEL_PATH = "/content/VibeVoice/models/VibeVoice-Realtime-0.5B"
  print("✅ Model loaded from local storage.")
else:
  MODEL_PATH = "microsoft/VibeVoice-Realtime-0.5B"

PROCESSOR = VibeVoiceStreamingProcessor.from_pretrained(MODEL_PATH)
MODEL = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cpu",
    attn_implementation="sdpa",
)
MODEL.eval()
MODEL.set_ddpm_inference_steps(num_steps=5)
VOICE_MAPPER = VoiceMapper()
print("✅ Model Ready.")

# ==========================================
# 4. GENERATION FUNCTION
# ==========================================
def generate_speech(text: str, speaker_name: str, cfg_scale: float = 1.5):
    if not text or not text.strip():
        return None, "❌ Please enter text."
    
    try:
        full_script = text.strip().replace("'", "'").replace('"', '"').replace('"', '"')
        voice_sample = VOICE_MAPPER.get_voice_path(speaker_name)
        
        if not voice_sample:
             return None, f"❌ Voice '{speaker_name}' not found."

        all_prefilled_outputs = torch.load(voice_sample, map_location="cuda", weights_only=False)

        inputs = PROCESSOR.process_input_with_cached_prompt(
            text=full_script,
            cached_prompt=all_prefilled_outputs,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        MODEL.to("cuda")
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to("cuda")

        start_time = time.time()
        with torch.cuda.amp.autocast():
            outputs = MODEL.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=PROCESSOR.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs) if all_prefilled_outputs else None,
            )
        generation_time = time.time() - start_time

        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            sample_rate = 24000
            audio_samples = outputs.speech_outputs[0].shape[-1] if len(outputs.speech_outputs[0].shape) > 0 else len(outputs.speech_outputs[0])
            audio_duration = audio_samples / sample_rate
            rtf = generation_time / audio_duration if audio_duration > 0 else float("inf")

            output_dir = "./outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"generated_{int(time.time())}.wav")

            PROCESSOR.save_audio(outputs.speech_outputs[0].cpu(), output_path=output_path)

            # --- Status Card (Neutral Dark Gray to work on any theme) ---
            status = f"""
            <div style='background-color: #333; color: white; padding: 20px; border-radius: 10px; margin-top: 10px;'>
                <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                    <div style='font-size: 20px; margin-right: 10px;'>✅</div>
                    <h3 style='margin: 0; color: white;'>Generation Complete</h3>
                </div>
                <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; font-size: 14px;'>
                    <div>⏱️ Time: <b>{generation_time:.2f}s</b></div>
                    <div>🔤 Total Characters: <b>{len(text)}</b></div>
                    <div>⚡ RTF: <b>{rtf:.2f}x</b></div>
                </div>
            </div>
            """
            
            MODEL.to("cpu")
            torch.cuda.empty_cache()
            return output_path, status
        else:
            MODEL.to("cpu")
            torch.cuda.empty_cache()
            return None, "❌ Error"

    except Exception as e:
        MODEL.to("cpu")
        torch.cuda.empty_cache()
        return None, f"❌ Error: {str(e)}"

# ==========================================
# 5. UI IMPLEMENTATION
# ==========================================
def ui():
  custom_css = """.gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"""

  with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
      gr.HTML("""
          <div style="text-align: center; margin: 20px auto; max-width: 800px;">
              <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎙️ VibeVoice-Realtime-0.5B </h1>
          </div>""")
      with gr.Row():
          with gr.Column():
              text_input = gr.Textbox(
                  label="Input Text",
                  placeholder="Enter text here...",
                  lines=6
              )
              
              with gr.Row():
                  speaker_dropdown = gr.Dropdown(
                      choices=list(VOICE_MAPPER.available_voices.keys()),
                      value=list(VOICE_MAPPER.available_voices.keys())[0] if VOICE_MAPPER.available_voices else None,
                      label="Speaker",
                      interactive=True
                  )
                  
                  cfg_slider = gr.Slider(
                      minimum=1.0, maximum=3.0, value=1.5, step=0.1,
                      label="CFG Scale"
                  )
              
              generate_btn = gr.Button("Generate Speech", variant="primary")

          with gr.Column():
              audio_output = gr.Audio(label="Generated Audio", type="filepath")
              status_output = gr.HTML()
      with gr.Accordion("Examples", open=False):
        gr.Examples(
            examples = [
      ["Good morning! I hope your day is off to a wonderful start.", "en-Carter_man", 1.5],        # cheerful
      ["Please remain calm. We are resolving the issue immediately.", "en-Davis_man", 1.4],       # serious / calm
      ["Oh my gosh! You won?! That's incredible!", "en-Emma_woman", 1.7],                         # excited
      ["I’m not sure that's a good idea… but I’ll trust your judgment.", "en-Frank_man", 1.5],    # hesitant
      ["Hi! Thank you for calling, how may I assist you today?", "en-Grace_woman", 1.6],           # polite / friendly
      ["What?! You’re telling me that actually worked?", "en-Mike_man", 1.5],                     # surprised
      ["Let me explain. Neural networks learn patterns from data over time.", "in-Samuel_man", 1.3] # informative
  ]
  ,
            inputs=[text_input, speaker_dropdown, cfg_slider],
            outputs=[audio_output, status_output],
            fn=generate_speech,
            run_on_click=False
        )

      generate_btn.click(
          fn=generate_speech,
          inputs=[text_input, speaker_dropdown, cfg_slider],
          outputs=[audio_output, status_output]
      )
  return demo
# demo=ui()
# demo.launch()

import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def run_app(share,debug):
    demo=ui()
    demo.queue().launch(share=share,debug=debug)
if __name__ == "__main__":
    run_app()

