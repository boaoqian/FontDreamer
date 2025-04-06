import os
import random

from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

DEFAULT_CHARSET  = list("国好想")

class FontsDataset(Dataset):
    def __init__(self, fonts_root, characters_set_path, img_size):
        self.characters_set = load_characters_set(characters_set_path)
        self.fonts_path = list(set(find_fonts(fonts_root)))
        self.fonts_path = [font for font in self.fonts_path if vaild_font(font, self.characters_set)]
        self.img_size = img_size
        self.size = len(self.fonts_path) * len(self.characters_set)
        self.sample_num = len(self.characters_set)
        self.fonts_random_step = len(self.fonts_path) -1
        self.char_random_step = self.sample_num -1


    def draw_char(self, font_path, char):
        img = Image.new("L", self.img_size, color="white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, self.img_size[1]-10)
        # center
        _,_,w,h = font.getbbox(char)
        draw.text(((self.img_size[0]-w)//2,(self.img_size[1]-h)//2), char, font=font, fill="black")
        return img

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        font_id, char_id = divmod(idx, self.sample_num)
        char = self.characters_set[char_id]
        pos_char = self.characters_set[random.randint(font_id+1, self.char_random_step+font_id)%self.sample_num]
        neg_font_id = random.randint(1, self.fonts_random_step)
        neg_char = self.characters_set[random.randint(0,self.char_random_step)]
        anchor = self.draw_char(self.fonts_path[font_id], char)
        pos = self.draw_char(self.fonts_path[font_id], pos_char)
        neg = self.draw_char(self.fonts_path[neg_font_id], neg_char)
        return anchor, pos, neg

def find_fonts(directory, extensions=('.ttf', '.otf', '.ttc')):
    font_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                font_files.append(os.path.join(root, file))
    return font_files

def load_characters_set(path):
    with open(path, 'r') as f:
        characters = f.read().replace("\n","")
        characters = list(characters)
    return characters

def vaild_font(font_path, charset):
    try:
        font = ImageFont.truetype(font_path, 72)
        for c in charset:
            font.getbbox(c)
        return True
    except Exception as e:
        print(e)
        print(font_path)
        return False
