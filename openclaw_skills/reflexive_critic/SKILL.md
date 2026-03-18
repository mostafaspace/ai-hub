# Skill: Reflexive Creative Critic 🧠🔄

This skill empowers the agent to self-audit its creative outputs (video, audio, images) by using multi-modal feedback loops. It allows the agent to "see" and "hear" its own generations to verify quality before presenting them to the user.

## API Endpoint

- **URL**: `http://{{HUB_IP}}:9000/v1/workflows/audit`
- **Method**: `POST`
- **Payload Structure**:
  ```json
  {
    "media_url": "string (URL to the generated asset)",
    "prompt_context": "string (the original prompt or context to check against)",
    "check_audio": true,
    "check_visual": true
  }
  ```

## Usage Pattern: The Reflexive Loop

When performing complex creative tasks (like the `Director` workflow), use this skill to ensure professional quality:

1.  **Generate**: Call the generation workflow (e.g., `POST /v1/workflows/director`).
2.  **Audit**: Pass the resulting `output_url` to `POST /v1/workflows/audit`.
3.  **Analyze**: 
    - Review the `audio_audit.transcript` for speech clarity, missed words, or robotic tone.
    - Review the `visual_audit` descriptions (sampled from across the video) to ensure visual consistency and alignment with the prompt.
4.  **Refine (Optional)**: If the audit reveals issues (e.g., "The image is too dark," "The speech skipped a sentence"), **re-run the generation** with a corrected prompt.
5.  **Deliver**: Only provide the asset to the user once it passes your internal qualitative audit.

## Example Quality Report Response

```json
{
  "task_id": "...",
  "media_url": "...",
  "audio_audit": {
    "text": "The quick brown fox jumps over the lazy dog."
  },
  "visual_audit": [
    { "timestamp": 0.5, "description": "A close up of a fox jumping in a forest." },
    { "timestamp": 2.5, "description": "A fox landing near a sleeping dog." }
  ],
  "overall_status": "COMPLETED"
}
```

## Prompting Tips for Auditing
- If the audio audit shows "gibberish" or repetitive words, increase the `temperature` or simplify the input text in the next run.
- If the visual audit descriptions don't match the subject, add "highly detailed, clear focus" to the next generation prompt.
