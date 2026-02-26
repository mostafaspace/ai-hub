import sys
import traceback

sys.path.append(r'D:\antigravity\ltx2_temp\packages\ltx-core\src')
sys.path.append(r'D:\antigravity\ltx2_temp\packages\ltx-pipelines\src')

try:
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    print("SUCCESS")
except Exception as e:
    traceback.print_exc()
