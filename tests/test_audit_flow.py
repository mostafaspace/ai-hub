import os
import sys
import unittest
from unittest.mock import MagicMock, patch, mock_open
import asyncio
from pydantic import BaseModel

# Add root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mocking modules that might not be in the test environment or have side effects
mock_fastapi = MagicMock()
sys.modules["fastapi"] = mock_fastapi
sys.modules["fastapi.responses"] = MagicMock()
sys.modules["fastapi.staticfiles"] = MagicMock()
sys.modules["fastapi.middleware.cors"] = MagicMock()

# Mock httpx
mock_httpx = MagicMock()
sys.modules["httpx"] = mock_httpx

# Import the code to test (we'll focus on the logic in run_audit_task)
# Since server.py has top-level execution code, we might need to mock more
from orchestrator.server import AuditRequest, run_audit_task

class TestAuditWorkflow(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.task_id = "test-task-123"
        self.req = AuditRequest(
            media_url="http://example.com/test.mp4",
            prompt_context="Testing",
            check_audio=True,
            check_visual=True
        )

    @patch("orchestrator.server.http_client")
    @patch("orchestrator.server.record_task")
    @patch("orchestrator.server.extract_audio_for_transcription")
    @patch("orchestrator.server.extract_thumbnail")
    @patch("orchestrator.server.detect_duration")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("orchestrator.server.BACKENDS", {"asr": "http://asr", "vision": "http://vision"})
    async def test_run_audit_task_success(self, mock_file, mock_makedirs, 
                                        mock_duration, mock_thumb, mock_audio_ext, 
                                        mock_record, mock_http):
        
        # Setup mocks
        mock_duration.return_value = 10.0
        mock_audio_ext.return_value = (True, "Success")
        mock_thumb.return_value = (True, "Success")
        
        # Mock Download
        mock_resp_dl = MagicMock()
        mock_resp_dl.status_code = 200
        mock_resp_dl.content = b"fake-media-content"
        
        # Mock ASR
        mock_resp_asr = MagicMock()
        mock_resp_asr.status_code = 200
        mock_resp_asr.json.return_value = {"text": "Transcribed text"}
        
        # Mock Vision
        mock_resp_vision = MagicMock()
        mock_resp_vision.status_code = 200
        mock_resp_vision.json.return_value = {
            "choices": [{"message": {"content": "Visual description"}}]
        }
        
        from unittest.mock import AsyncMock
        mock_http.get = AsyncMock(return_value=mock_resp_dl)
        mock_http.post = AsyncMock(side_effect=[mock_resp_asr, mock_resp_vision, mock_resp_vision, mock_resp_vision])
        
        # Run
        await run_audit_task(self.task_id, self.req)
        
        # Verify
        mock_record.assert_any_call(self.task_id, "content_auditor", "COMPLETED")
        self.assertTrue(mock_http.post.called)
        
        # Check that report.json was written
        # It should be in work_dir which is outputs/audit/test-task-123/report.json
        actual_calls = [call[0][0] for call in mock_file.call_args_list if "report.json" in call[0][0]]
        self.assertTrue(len(actual_calls) > 0, "report.json was not written")

if __name__ == "__main__":
    asyncio.run(unittest.main())
