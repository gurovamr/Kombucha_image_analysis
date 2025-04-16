import os
import cv2

# === Load templates ===
template_paths = [
    "data/Experiment_1/cropping_templates/cam0_04-04_17-00(1).jpg",
    "data/Experiment_1/cropping_templates/cam0_04-04_18-00(2).jpg",
    "data/Experiment_1/cropping_templates/cam0_04-05_01-00(9).jpg",
    "data/Experiment_1/cropping_templates/cam0_04-05_13-00(21).jpg",
    "data/Experiment_1/cropping_templates/cam0_04-06_14-00(46).jpg",
    "data/Experiment_1/cropping_templates/cam0_04-07_08-00(64).jpg",
    "data/Experiment_1/cropping_templates/cam0_04-08_14-00(94).jpg",
    "data/Experiment_1/cropping_templates/cam0_04-08_23-00(103).jpg",
    "data/Experiment_1/cropping_templates/cam0_04-11_13-00(164).jpg",
    "data/Experiment_1/cropping_templates/cam0_04-15_02-00(249).jpg"
]
templates = [(cv2.imread(p, 0), p) for p in template_paths]  # (template_image, path)

# === Output and input folders ===
input_folder = "data/Experiment_1/exp1_cam0"
output_folder = "data/Experiment_1/cropped_cam0"
os.makedirs(output_folder, exist_ok=True)

image_paths = sorted([
    os.path.join(input_folder, f)
    for f in os.listdir(input_folder)
    if f.lower().endswith(".jpg")
])

for img_path in image_paths:
    img_color = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    best_score = -1
    best_crop = None
    best_template = None

    for template, template_path in templates:
        h, w = template.shape
        result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            top_left = max_loc
            crop = img_color[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
            best_score = max_val
            best_crop = crop
            best_template = os.path.basename(template_path)

    # === Save the best crop if above threshold ===
    if best_score >= 0.6:
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, best_crop)
        print(f"✅ Saved {filename} (matched: {best_template}, score: {best_score:.2f})")
    else:
        print(f"❌ Skipped {img_path} — low match score: {best_score:.2f}")
