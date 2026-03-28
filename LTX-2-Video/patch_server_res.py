import os

with open("server.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace all occurrences of Form(704) and Form(1280)
content = content.replace("Form(704)", "Form(576)")
content = content.replace("Form(1280)", "Form(1024)")

# Ensure defaults are also set in the Pydantic model
content = content.replace("default=704", "default=576")
content = content.replace("default=1280", "default=1024")

with open("server.py", "w", encoding="utf-8") as f:
    f.write(content)

print("server.py updated to 1024x576.")
