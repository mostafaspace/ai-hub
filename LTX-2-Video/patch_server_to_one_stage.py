import os

with open("server.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # 1. Pipeline change
    if "self.pipeline = TI2VidTwoStagesPipeline(" in line:
        new_lines.append(line.replace("TI2VidTwoStagesPipeline(", "TI2VidOneStagePipeline("))
        continue
    # 2. Argument removal for One-Stage
    if "distilled_lora=distilled_lora_config or []," in line:
        continue
    if "spatial_upsampler_path=SPATIAL_UPSAMPLER," in line:
        new_lines.append("                    device=DEVICE,\n")
        continue
    # 3. Default steps
    if "DEFAULT_NUM_INFERENCE_STEPS = 55" in line: # I already changed this
        new_lines.append(line)
        continue
    if "DEFAULT_NUM_INFERENCE_STEPS = 35" in line:
        new_lines.append("DEFAULT_NUM_INFERENCE_STEPS = 75\n")
        continue
    
    new_lines.append(line)

with open("server.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("server.py patched for One-Stage HD.")
