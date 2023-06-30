from PIL import Image
import glob
from PIL import GifImagePlugin
GifImagePlugin.LOADING_STRATEGY = GifImagePlugin.LoadingStrategy.RGB_AFTER_DIFFERENT_PALETTE_ONLY

# Create the frames
frames = []
imgs = glob.glob("figures/*.png")
imgs.sort(key=lambda x: int(x.split('/')[1].split('.')[0]))
for i in imgs:
    new_frame = Image.open(i).convert('P')
    frames.append(new_frame)

for _ in range(10):
    frames.append(frames[-1])

# Save into a GIF file that loops forever
frames[0].save('grid_interpolation.gif', format='GIF',
               append_images=frames[1:],
               save_all=True, quality=100,
               duration=100, loop=0)