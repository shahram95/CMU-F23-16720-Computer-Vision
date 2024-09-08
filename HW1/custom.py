import random

from PIL import Image, ImageDraw
import os

# Function to read lines from a file into a list
def read_lines_from_file(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Read train image file paths
train_files = read_lines_from_file('../data/train_files.txt')

# Read train labels
train_labels = read_lines_from_file('../data/train_labels.txt')

# Combine file paths and labels into a list of tuples
combined = list(zip(train_files, train_labels))

# Create a dictionary to hold file paths by class
class_dict = {}
for path, label in combined:
    if label not in class_dict:
        class_dict[label] = []
    class_dict[label].append(path)

# Randomly sample 25 file paths from each class and create a final list of tuples
final_list = []
for label, paths in class_dict.items():
    sampled_paths = random.sample(paths, 5)  # Assuming each class has at least 25 samples
    final_list += [(path, int(label)) for path in sampled_paths]

# Shuffle the final list to randomize the order of classes
#random.shuffle(final_list)

print(final_list)

'''
# To verify sampling
cnt_dict = dict()
for _,label in final_list:
    cnt_dict[label] = cnt_dict.get(label,0) + 1

print(cnt_dict)
'''
augmented_files = []
augmented_labels = []

output_dir = "../data/augmented"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for img_path, label in final_list:
    # Open the image
    img_path = os.path.join("../data",img_path)
    img = Image.open(img_path)
    if img.mode == 'L':
        img = Image.merge("RGB", (img, img, img))  # Stack the single channel to make it RGB
    
    # Get the dimensions of the image
    width, height = img.size
    
    # Calculate the size of each grid cell
    grid_w = width // 16
    grid_h = height // 16
    
    # Initialize ImageDraw object
    draw = ImageDraw.Draw(img)

    # Loop through the 4x4 grid
    for i in range(16):
        for j in range(16):
            x1 = i * grid_w
            y1 = j * grid_h
            x2 = (i + 1) * grid_w
            y2 = (j + 1) * grid_h
            
            # Randomly decide to black out the patch
            random_chance = random_number = random.randint(1, 100)
            if random_chance<=20:
                draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
            
            
                
    # Generate the output path
    base_name = os.path.basename(img_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name}_aug{ext}")
    
    # Save the relative path and label for augmented files
    relative_path = output_path.replace("..\data", "")
    augmented_files.append(relative_path.split("/")[-1])
    augmented_labels.append(label)

    # Save the modified image
    img.save(output_path)

combined_files = train_files + augmented_files
combined_labels = train_labels + augmented_labels

# Write to new files
with open('../data/train_files_aug.txt', 'w') as f:
    for item in combined_files:
        f.write(f"{item}\n")

with open('../data/train_labels_aug.txt', 'w') as f:
    for item in combined_labels:
        f.write(f"{item}\n")