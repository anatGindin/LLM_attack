from PIL import Image, PngImagePlugin

# Create a blank image
img = Image.new("RGB", (200, 200), color="white")

# Add the malicious code as metadata
meta = PngImagePlugin.PngInfo()
meta.add_text("Comment", 'def hidden(): print("Hello World")\nhidden()')

# Save image with the code embedded
img.save("payload.png", pnginfo=meta)
