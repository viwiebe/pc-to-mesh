import cv2
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Straighten building outlines in color heatmap using HSV masking.")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("--output", default="cleaned_buildings.png", help="Output image path")
    parser.add_argument("--epsilon", type=float, default=0.02, help="Simplification factor for contours")
    args = parser.parse_args()

    # Load image
    color_img = cv2.imread(args.input)
    if color_img is None:
        print("Error loading image.")
        return

    # Convert to HSV color space
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

    # Define red/orange hue range (tune if needed)
    lower1 = np.array([0, 100, 100])
    upper1 = np.array([15, 255, 255])
    lower2 = np.array([160, 100, 100])
    upper2 = np.array([179, 255, 255])

    # Combine masks for red/orange
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Optional: remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours and simplify them
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_mask = np.zeros_like(mask)

    for cnt in contours:
        epsilon = args.epsilon * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(cleaned_mask, [approx], -1, 255, thickness=cv2.FILLED)

    # Combine cleaned mask with original image
    buildings = cv2.bitwise_and(color_img, color_img, mask=cleaned_mask)
    background = cv2.bitwise_and(color_img, color_img, mask=cv2.bitwise_not(cleaned_mask))
    result = cv2.add(buildings, background)

    # Save result
    cv2.imwrite(args.output, result)
    print(f"Saved cleaned image to: {args.output}")

if __name__ == "__main__":
    main()
