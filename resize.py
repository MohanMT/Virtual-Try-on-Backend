# from PIL import Image
# import os

# # Open the image
# img = Image.open('Images/new1.jpg')

# # Target dimensions
# target_width = 320
# target_height = 344
# target_aspect = target_width / target_height

# # Get original dimensions
# original_width, original_height = img.size
# original_aspect = original_width / original_height

# # Crop the image to match the target aspect ratio
# if original_aspect > target_aspect:
#     # Wider than target, crop width
#     new_width = int(original_height * target_aspect)
#     left = (original_width - new_width) // 2
#     right = left + new_width
#     top = 0
#     bottom = original_height
# else:
#     # Taller than target, crop height
#     new_height = int(original_width / target_aspect)
#     top = (original_height - new_height) // 2
#     bottom = top + new_height
#     left = 0
#     right = original_width

# # Perform the crop
# img = img.crop((left, top, right, bottom))

# # Resize the cropped image to the target dimensions
# img = img.resize((target_width, target_height), Image.LANCZOS)

# # Save the final image
# output_folder = "Resized"
# os.makedirs(output_folder, exist_ok=True)
# img.save(os.path.join(output_folder, 'resized_image.png'))

# # Output sizes for verification
# print("Final image size:", img.size)  # This should be the exact target size
# img.show()


from PIL import Image
img=Image.open('Images/new1.jpg')
print(img.width,img.height)
img.show()
img=img.resize((int(img.width*2),int(img.height*2)),resample=Image.LANCZOS)
print(img.width,img.height)
img.show()
