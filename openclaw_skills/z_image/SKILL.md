---
name: z_image
description: Z-Image Vision Generator - Generate images from text prompts using an async polling queue.
---

# Z-Image Vision API Skill

This API uses a background task queue. Generating images takes a few minutes, so you MUST use this 2-step polling process. DO NOT use the synchronous generation endpoints.

## Server Details

-   **Base URL**: `http://192.168.1.26:8003`

## 2-Step Agent Workflow

### Step 1: Start a Task
You have two abilities here: Generating *new* images, or Editing *existing* images.

#### Option A: Generate a New Image
Send a simple POST request to queue the prompt.

**POST** `/v1/images/async_generate`

**JSON Body:**
```json
{
  "prompt": "A happy cat sitting on a rug, highly detailed, photorealistic"
}
```

#### Option B: Edit an Existing Image
If the user provides an image and asks you to change it (e.g. "Make the cat sad"), you must send the image to the Edit endpoint as a `multipart/form-data` request.

**POST** `/v1/images/async_edit`

**FormData Body:**
- `image`: The file bytes (e.g., `open('cat.png', 'rb')`)
- `prompt`: The editing instruction (e.g., "Make the cat sad")

#### Task Response (For both options)
```json
{
  "task_id": "8b9cad0e1f20",
  "status": "processing"
}
```

### Step 2: Poll for Result 
Wait 5 seconds, then repeatedly call this endpoint using the `task_id` you received above until the status is `completed`.

**GET** `/v1/images/tasks/{task_id}`

**Response while rendering:**
```json
{
  "status": "processing"
}
```

**Final Response (when done):**
```json
{
  "status": "completed",
  "data": [
      {
          "url": "http://192.168.1.26:8003/outputs/gen_...png"
      }
  ]
}
```
Once you get `"status": "completed"`, you have successfully generated the image! Read the `url` from the `data` array and provide it to the user.

## Delivering as a WhatsApp Image (File Attachment)
If you need to send the actual image file to a user (e.g., via a WhatsApp messaging node) instead of the local HTTP link, you have two options to get the file data:

**Option 1: Download the File Bytes (Easiest)**
Using the `url` returned in Step 2, simply make a standard HTTP `GET` request to it (e.g., `http://192.168.1.26:8003/outputs/gen_...png`). The response body will be the raw `image/png` binary file. Save it to a local temporary file and attach it to your message.

**Option 2: Request Base64 JSON**
If your tool natively accepts Base64 strings, you can add `"response_format": "b64_json"` to your initial `POST` payload in Step 1. The final polled result will contain a `b64_json` field with the Base64-encoded PNG string instead of a `url`.

## Pre-flight & Health

If you want to check if the server is online before starting:
- **GET** `/health`
- Returns `{"status": "ok"}` if the server is reachable.
