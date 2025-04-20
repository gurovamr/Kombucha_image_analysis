import os
import cv2

# === Load templates with debug logging ===
template_paths = [
    "data/Experiment_1/cropping_templates/cam1/cam1_04-04_17-00(1).jpg",
    "data/Experiment_1/cropping_templates/cam1/cam1_04-04_18-00(2).jpg",
    "data/Experiment_1/cropping_templates/cam1/cam1_04-05_06-00(14).jpg",
    "data/Experiment_1/cropping_templates/cam1/cam1_04-05_15-00(23).jpg",
    "data/Experiment_1/cropping_templates/cam1/cam1_04-06_13-00(45).jpg",
    "data/Experiment_1/cropping_templates/cam1/cam1_04-06_18-00(50).jpg",
    "data/Experiment_1/cropping_templates/cam1/cam1_04-07_05-00(61).jpg",
    "data/Experiment_1/cropping_templates/cam1/cam1_04-07_11-00(67).jpg",
    "data/Experiment_1/cropping_templates/cam1/cam1_04-08_16-00(96).jpg",
    "data/Experiment_1/cropping_templates/cam1/cam1_04-08_17-00(97).jpg",
    "data/Experiment_1/cropping_templates/cam1/cam1_04-09_06-00(110).jpg",
    "data/Experiment_1/cropping_templates/cam1/cam1_04-11_14-00(164).jpg"
]

templates = []
for p in template_paths:
    print(f"🔍 Checking template path: {p}")
    if not os.path.exists(p):
        print(f"❌ File does not exist: {p}")
        continue

    img = cv2.imread(p, 0)
    if img is None:
        print(f"⚠️ Failed to load image (check format or corruption): {p}")
        continue

    templates.append((img, p))
    print(f"✅ Loaded template: {p}")

if not templates:
    print("⛔ No valid templates loaded. Exiting.")
    exit()

# === Output and input folders ===
input_folder = "data/Experiment_1/exp1_cam1"
output_folder = "data/Experiment_1/cropped_cam1"
os.makedirs(output_folder, exist_ok=True)

image_paths = sorted([
    os.path.join(input_folder, f)
    for f in os.listdir(input_folder)
    if f.lower().endswith(".jpg")
])

for img_path in image_paths:
    print(f"\n📷 Processing image: {img_path}")
    img_color = cv2.imread(img_path)

    if img_color is None:
        print(f"⚠️ Failed to read image: {img_path}")
        continue

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    best_score = -1
    best_crop = None
    best_template = None

    for template, template_path in templates:
        if template is None:
            print(f"⛔ Skipping null template: {template_path}")
            continue

        try:
            h, w = template.shape
            result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_score:
                top_left = max_loc
                crop = img_color[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
                best_score = max_val
                best_crop = crop
                best_template = os.path.basename(template_path)
        except Exception as e:
            print(f"❗ Error processing template {template_path}: {e}")
            continue

    if best_score >= 0.6 and best_crop is not None:
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, best_crop)
        print(f"✅ Saved {filename} (matched: {best_template}, score: {best_score:.2f})")
    else:
        print(f"❌ Skipped {img_path} — low match score: {best_score:.2f}")
