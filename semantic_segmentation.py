import cv2
import supervision as sv
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import clip
from PIL import Image

# ===========================
# 1. Define Ontology
# ===========================

# Define your ontology without 'walls' and 'ship', and with additional class 'yacht'
ontology = CaptionOntology({
    "building": "building",
    "grass": "grass",
    "plants": "plants",
    "trees": "trees",
    "hedge": "hedge",
    "river": "river",
    "sky": "sky",
    "footpath": "footpath",
    "pavement": "pavement",
    "fence": "fence",
    "house": "house",
    "shrubs": "shrubs",
    "yacht": "yacht",
    "greenery": "greenery"
})

# Initialize Grounded SAM with the updated ontology
base_model = GroundedSAM(ontology=ontology)

# ===========================
# 2. Load Image
# ===========================

# Load your image
image_path = "Desktop/ImageSegmentation/Expl1.png"  # Update this with your image path
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Image not found at {image_path}")

# ===========================
# 3. Predict Segmentation Masks
# ===========================

# Predict segmentation masks
detections = base_model.predict(image_path)

# Debugging: Print the detections
print("Detections:", detections)

# ===========================
# 4. Define Color Map
# ===========================

# Define a color map for each class with specified colors
# Colors are in RGB format and uniquely assigned
color_map = {
    "building": (128, 128, 128),    # Dark Grey
    "grass": (0, 255, 0),           # Green
    "plants": (0, 200, 0),          # Medium Green
    "trees": (0, 150, 0),           # Darker Green
    "hedge": (34, 139, 34),         # Forest Green
    "river": (0, 0, 255),           # Ocean Blue
    "sky": (128, 0, 128),           # Purple
    "footpath": (211, 211, 211),    # Light Gray
    "pavement": (211, 211, 211),    # Light Gray
    "fence": (255, 0, 0),           # Red
    "house": (128, 128, 128),       # Dark Grey
    "shrubs": (0, 255, 0),          # Green
    "yacht": (255, 215, 0)          # Gold
}

# ===========================
# 5. Map Class IDs to Labels
# ===========================

# Map class IDs to labels based on the updated ontology
id_to_label = [
    "building",    # class_id = 0
    "grass",       # class_id = 1
    "plants",      # class_id = 2
    "trees",       # class_id = 3
    "hedge",       # class_id = 4
    "river",       # class_id = 5
    "sky",         # class_id = 6
    "footpath",    # class_id = 7
    "pavement",    # class_id = 8
    "fence",       # class_id = 9
    "house",       # class_id = 10
    "shrubs",      # class_id = 11
    "yacht"        # class_id = 12
]

# ===========================
# 6. Integrate OpenAI CLIP for Enhanced Detection
# ===========================

# Initialize CLIP model and preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Define textual prompts for each class
text_prompts = [
    "a photo of a multi-story building with windows",                        # building
    "a photo of green grass, and small green objects, very small dark green objects",                                               # grass
    "a photo of various plants all green colour shades pixels in the image",                                            # plants
    "a photo of dense forest with tall trees separated trees single short trees and green objects small dark green trees ",  # trees
    "a photo of a neat hedge",                                              # hedge
    "a photo of a flowing river",                                           # river
    "a photo of a clear purple sky with minimal clouds",                     # sky
    "a photo of a paved footpath",                                          # footpath
    "a photo of a yellow construction site"
    "a photo of a paved pavement area",                                     # pavement
    "a photo of a wooden fence",                                            # fence
    "a photo of a residential house and construction building",             # house
    "a photo of lush green shrubs small dark green shrub",                                         # shrubs
    "a photo of a sleek yacht sailing and docked", 
]

# Tokenize the prompts
text_tokens = clip.tokenize(text_prompts).to(device)

# Convert OpenCV image (BGR) to PIL image (RGB)
pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
clip_image = clip_preprocess(pil_image).unsqueeze(0).to(device)

# Encode image and text
with torch.no_grad():
    image_features = clip_model.encode_image(clip_image)
    text_features = clip_model.encode_text(text_tokens)

    # Normalize the features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    similarity = (image_features @ text_features.T).squeeze(0)  # Shape: (num_classes,)

# Define a similarity threshold
similarity_threshold = 0.15  # Adjusted for better recall

# ===========================
# 7. Filter Detections Based on CLIP Similarity with Priority Handling
# ===========================

# Define categories in order of priority to handle overlapping masks
# Higher priority categories are processed first
priority_categories = ["river", "yacht", "building", "house", "fence", "footpath", "pavement", "green", "sky"]

# Initialize cumulative mask to keep track of already assigned pixels
cumulative_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)

# Initialize lists to hold filtered masks and class IDs
filtered_masks = []
filtered_class_ids = []

for category in priority_categories:
    # Get class IDs for the current category
    if category == "green":
        # "green" includes multiple labels
        class_ids = [i for i, label in enumerate(id_to_label) if label in ["grass", "plants", "trees", "hedge", "shrubs"]]
    else:
        # Single label categories
        class_ids = [i for i, label in enumerate(id_to_label) if label == category]
    
    for class_id in class_ids:
        # Check if class_id is within bounds
        if class_id >= len(id_to_label):
            print(f"Warning: class_id {class_id} is out of bounds. Skipping.")
            continue
        
        label = id_to_label[class_id]
        
        # Get corresponding similarity score
        sim_score = similarity[class_id].item()
        
        if sim_score > similarity_threshold:
            # Find all detections with this class_id
            indices = np.where(detections.class_id == class_id)[0]
            for idx in indices:
                mask = detections.mask[idx]
                
                # Exclude already assigned areas
                exclusive_mask = np.logical_and(mask, ~cumulative_mask)
                
                if exclusive_mask.any():
                    filtered_masks.append(exclusive_mask)
                    filtered_class_ids.append(class_id)
                    
                    # Update cumulative mask
                    cumulative_mask = np.logical_or(cumulative_mask, exclusive_mask)
                    
                    print(f"Detection {idx} ({label}) retained with similarity {sim_score:.2f}")
                else:
                    print(f"Detection {idx} ({label}) has no exclusive area after applying priority masking. Skipping.")
        else:
            print(f"Detection with class_id {class_id} ({label}) has similarity {sim_score:.2f} below threshold. Skipping.")

# Create a new detections object with filtered results
class Detection:
    def __init__(self, mask, class_id):
        self.mask = mask
        self.class_id = class_id

if filtered_masks:
    # Convert list of masks to a numpy array
    filtered_masks_np = np.array(filtered_masks)
    filtered_class_ids_np = np.array(filtered_class_ids)
    filtered_detections = Detection(mask=filtered_masks_np, class_id=filtered_class_ids_np)
else:
    # Handle the case where no detections pass the threshold
    filtered_detections = Detection(mask=np.array([]), class_id=np.array([]))
    print("No detections passed the similarity threshold.")

# ===========================
# 8. Apply Custom Masks
# ===========================

def apply_custom_masks(image, detections, color_map, id_to_label, alpha=0.5):
    """
    Overlays colored masks on the image based on detections.

    :param image: Original image in BGR format.
    :param detections: Detections object containing masks and class_ids.
    :param color_map: Dictionary mapping labels to RGB colors.
    :param id_to_label: List mapping class_id to label.
    :param alpha: Transparency factor for mask overlay.
    :return: Annotated image with colored masks.
    """
    annotated_image = image.copy()
    num_detections = detections.class_id.shape[0]

    for i in range(num_detections):
        class_id = detections.class_id[i]
        if class_id >= len(id_to_label):
            print(f"Warning: class_id {class_id} is out of bounds. Skipping.")
            continue
        label = id_to_label[class_id]

        if label not in color_map:
            print(f"Warning: label '{label}' not in color_map. Skipping.")
            continue

        mask = detections.mask[i]
        color = color_map[label][::-1]  # Convert RGB to BGR for OpenCV

        # Create a colored mask
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[:] = color

        # Ensure mask is of type bool
        mask_bool = mask.astype(bool)

        # Apply the mask with transparency
        annotated_image[mask_bool] = cv2.addWeighted(
            annotated_image, 1 - alpha, colored_mask, alpha, 0
        )[mask_bool]

    return annotated_image

# ===========================
# 9. Add Legend
# ===========================

def add_legend(image, color_map, position=(10, 30)):
    """
    Adds a legend to the image.

    :param image: Image in BGR format.
    :param color_map: Dictionary mapping labels to RGB colors.
    :param position: Starting position for the legend.
    :return: Image with legend.
    """
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    box_height = 20
    box_width = 20
    spacing = 10

    for label, color in color_map.items():
        # Draw color box
        cv2.rectangle(
            image,
            (x, y),
            (x + box_width, y + box_height),
            color[::-1],  # Convert to BGR
            -1
        )
        # Put label text
        cv2.putText(
            image,
            label.replace("_", " ").title(),  # Format label text
            (x + box_width + spacing, y + box_height - 5),
            font,
            font_scale,
            (255, 255, 255),  # White text
            font_thickness,
            cv2.LINE_AA
        )
        y += box_height + spacing  # Move to next line

    return image

# ===========================
# 10. Calculate and Annotate Percentages
# ===========================

def calculate_and_annotate_percentage(image, detections, id_to_label, target_labels, category_name, cumulative_mask=None):
    """
    Calculates the percentage area of target labels and annotates it on the mask image.

    :param image: Original image in BGR format.
    :param detections: Detections object containing masks and class_ids.
    :param id_to_label: List mapping class_id to label.
    :param target_labels: List of labels to calculate percentage for.
    :param category_name: Name of the category (e.g., "Green", "Building", "River").
    :param cumulative_mask: Boolean mask of already processed areas to exclude overlaps.
    :return: Annotated mask image with percentage and the combined mask.
             Also returns target_percentage.
    """
    # Initialize a combined mask for the target labels
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)

    for i in range(len(detections.class_id)):
        class_id = detections.class_id[i]
        if class_id >= len(id_to_label):
            continue
        label = id_to_label[class_id]
        if label in target_labels:
            mask = detections.mask[i]
            if cumulative_mask is not None:
                mask = np.logical_and(mask, ~cumulative_mask)  # Exclude already assigned areas
            combined_mask = np.logical_or(combined_mask, mask)  # Combine masks using logical OR

    # Calculate the number of target pixels
    target_pixels = np.sum(combined_mask)

    # Calculate the total number of pixels in the image
    total_pixels = image.shape[0] * image.shape[1]

    # Calculate the percentage of target area
    target_percentage = (target_pixels / total_pixels) * 100

    print(f"\nPercentage of {category_name.lower()} in the image: {target_percentage:.2f}%")
    
    # Create a colored mask for visualization
    if category_name.lower() == "green":
        color = (0, 255, 0)  # Pure Green
    elif category_name.lower() in ["building", "house"]:
        color = (128, 128, 128)  # Dark Grey
    elif category_name.lower() == "river":
        color = (0, 0, 255)  # Ocean Blue
    elif category_name.lower() in ["footpath", "pavement"]:
        color = (211, 211, 211)  # Light Gray
    elif category_name.lower() == "fence":
        color = (255, 0, 0)  # Red
    elif category_name.lower() == "yacht":
        color = (255, 215, 0)  # Gold
    elif category_name.lower() == "sky":
        color = (128, 0, 128)  # Purple
    else:
        color = (255, 255, 255)  # White as default

    mask_visual = np.zeros_like(image, dtype=np.uint8)
    mask_visual[:] = (0, 0, 0)  # Start with black

    mask_visual[combined_mask] = color  # Apply the color to target areas

    # Add the percentage text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text = f"{category_name} Area: {target_percentage:.2f}%"
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = 10
    text_y = text_size[1] + 10

    # Draw a semi-transparent rectangle behind the text for better visibility
    rectangle_bgr = (0, 0, 0)  # Black rectangle
    cv2.rectangle(
        mask_visual,
        (text_x - 5, text_y - text_size[1] - 5),
        (text_x + text_size[0] + 5, text_y + 5),
        rectangle_bgr,
        -1
    )

    # Add text
    cv2.putText(
        mask_visual,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),  # White text
        font_thickness,
        cv2.LINE_AA
    )

    return mask_visual, combined_mask, target_percentage

# ===========================
# 11. Create Output Directory
# ===========================

# Create Output Directory if Not Exists
output_dir = "output_image1_new"
os.makedirs(output_dir, exist_ok=True)

# ===========================
# 12. Apply Masks and Calculate Percentages with Priority Handling
# ===========================

# Apply custom colored masks using filtered detections
annotated_frame = apply_custom_masks(image, filtered_detections, color_map, id_to_label, alpha=0.5)

# Add legend to the annotated image
annotated_frame = add_legend(annotated_frame, color_map)

# Save the Annotated Segmented Image with Legend
segmented_image_path = os.path.join(output_dir, "segmented_image.png")
cv2.imwrite(segmented_image_path, annotated_frame)
print(f"\nSegmented image saved at: {segmented_image_path}")

# ===========================
# 13. Calculate and Annotate Percentages for Each Category
# ===========================

# Define target labels for each category with priority order
categories = [
    ("River", ["river"]),
    ("Yacht", ["yacht"]),
    ("Building", ["building"]),
    ("House", ["house"]),
    ("Fence", ["fence"]),
    ("Footpath", ["footpath", "pavement"]),
    ("Green", ["grass", "plants", "trees", "hedge", "shrubs"]),
    ("Sky", ["sky"])
]

# Dictionary to hold paths of percentage mask images
percentage_mask_paths = {}

# Initialize cumulative mask for percentage calculations
cumulative_mask_percentage = np.zeros((image.shape[0], image.shape[1]), dtype=bool)

# Initialize a dictionary to store percentage values
percentage_values = {}

# Optional: Define a minimum mask area threshold to avoid reporting very small regions
MIN_MASK_AREA = 50  # Adjust based on your image resolution and object sizes

for category_name, target_labels in categories:
    mask_visual, combined_mask, target_percentage = calculate_and_annotate_percentage(
        image, filtered_detections, id_to_label, target_labels, category_name, cumulative_mask=cumulative_mask_percentage
    )
    # Check if the mask meets the minimum area requirement
    if np.sum(combined_mask) < MIN_MASK_AREA:
        print(f"{category_name} mask has a small area ({np.sum(combined_mask)} pixels). Skipping annotation.")
        continue
    # Define file name based on category
    mask_filename = f"{category_name.lower()}_percentage_mask.png"
    mask_path = os.path.join(output_dir, mask_filename)
    # Save the mask image
    cv2.imwrite(mask_path, mask_visual)
    print(f"{category_name} percentage mask image saved at: {mask_path}")
    # Update cumulative mask
    cumulative_mask_percentage = np.logical_or(cumulative_mask_percentage, combined_mask)
    # Store the percentage value
    percentage_values[category_name] = round(float(target_percentage), 2)
    # Store the path if needed later
    percentage_mask_paths[category_name] = mask_path

# ===========================
# 14. Calculate Undetected Percentage
# ===========================

# Calculate the sum of all detected percentages
total_detected_percentage = sum(percentage_values.values())

# Calculate undetected percentage
undetected_percentage = round(100.0 - total_detected_percentage, 2)

print(f"\nAll detected categories sum up to: {total_detected_percentage:.2f}%")
print(f"Undetected region in the image: {undetected_percentage:.2f}%")

# ===========================
# 15. Save Summary of Percentages
# ===========================

# Create a summary text file
summary_text = ""
for category, percentage in percentage_values.items():
    summary_text += f"Percentage of {category.lower()} in the image: {percentage}%\n"
    summary_text += f"{category} percentage mask image saved at: output_images\\{category.lower()}_percentage_mask.png\n\n"
summary_text += f"Undetected region in the image: {undetected_percentage}%"

# Save the summary to a text file
summary_file_path = os.path.join(output_dir, "summary.txt")
with open(summary_file_path, "w") as f:
    f.write(summary_text)

print(f"\nSummary of percentages saved at: {summary_file_path}")

# ===========================
# 16. Final Notes
# ===========================

# ===========================


print("\nAll images and summary have been successfully saved in the 'output_images' directory.")
