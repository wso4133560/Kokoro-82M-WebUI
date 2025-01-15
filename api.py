#For client side
from gradio_client import Client
import shutil
import os
os.makedirs("temp_audio", exist_ok=True)
from gradio_client import Client
client = Client("http://127.0.0.1:7860/")
result = client.predict(
		text="Hello!!",
		model_name="kokoro-v0_19.pth",
		voice_name="af_bella",
		speed=1,
		trim=0,
		pad_between_segments=0,
		remove_silence=False,
		minimum_silence=0.05,
		api_name="/text_to_speech"
)

save_at=f"./temp_audio/{os.path.basename(result)}"
shutil.move(result, save_at)
print(f"Saved at {save_at}")
