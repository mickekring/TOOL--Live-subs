# script för att testa om openCV funkar.
# Skapar ett fönster och väntar på dig
# att trycka på valfri tangent för att stänga fönstret.
import cv2
cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL)
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()