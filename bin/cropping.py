import os
import cv2

def crop_batch(img_paths, batch_num=None):
    if not img_paths:
        print("Empty batch.")
        return

    first_img_path = img_paths[0]
    first_img = cv2.imread(first_img_path)

    # === Resize image for display ===
    scale = 0.1  # 50% of original size for ROI selection
    display_img = cv2.resize(first_img, (0, 0), fx=scale, fy=scale)

    # === ROI selection ===
    window_title = f"Select ROI for {os.path.basename(first_img_path)} (Batch {batch_num})" if batch_num else "Select ROI"
    x_disp, y_disp, w_disp, h_disp = cv2.selectROI(window_title, display_img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_title)

    if w_disp == 0 or h_disp == 0:
        print("Invalid ROI selected. Skipping batch.")
        return

    # === Scale ROI back to original size ===
    x = int(x_disp / scale)
    y = int(y_disp / scale)
    w = int(w_disp / scale)
    h = int(h_disp / scale)
    print(f"Selected ROI (batch {batch_num}): x={x}, y={y}, w={w}, h={h}")

    # === Preview crop from original image ===
    preview_crop = first_img[y:y + h, x:x + w]
    #cv2.imshow("Cropped Preview", preview_crop)
    #print("Press any key to confirm, or ESC to skip this batch.")
    preview_resized = cv2.resize(preview_crop, (0, 0), fx=0.1, fy=0.1)  # or 0.1
    cv2.imshow("Cropped Preview", preview_resized)

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == 27:  # ESC key
        print("Skipped batch.")
        return

    # === Apply crop to all images in batch ===
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read: {img_path}")
            continue

        cropped = img[y:y + h, x:x + w]

        # === Determine output folder ===
        if "cam0" in img_path:
            output_dir = "data/Experiment_1/cropped_cam0"
        elif "cam1" in img_path:
            output_dir = "data/Experiment_1/cropped_cam1"
        else:
            output_dir = "data/Experiment_1/cropped_misc"

        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, filename)

        # === Skip if already exists ===
        if os.path.exists(save_path):
            print(f"File exists, skipping: {filename}")
            continue

        cv2.imwrite(save_path, cropped)
        print(f"Cropped and saved: {filename} → {output_dir}")

if __name__ == "__main__":
    # === Input folder ===
    folder = "data/Experiment_1/exp1_cam0"
    files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".jpg")
    ])

    # === Define batch size ===
    batch_size = 10  # Change based on your workflow

    # === Process in batches ===
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        crop_batch(batch, batch_num=(i // batch_size + 1))
