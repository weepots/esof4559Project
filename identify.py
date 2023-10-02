import face_recognition
from PIL import Image, ImageDraw


image_of_bill = face_recognition.load_image_file('billGates1.png')
bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]


image_of_steve = face_recognition.load_image_file('steveJobs.png')
steve_face_encoding = face_recognition.face_encodings(image_of_steve)[0]


# create array of encodings and names
known_face_encodings = [
  bill_face_encoding,
  steve_face_encoding
]

known_face_names = [
  'Bill Gates',
  'Steve Jobs'
]

test_image = face_recognition.load_image_file('bill_steve.png')

# find faces in test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

#convert to PIL format
pil_image = Image.fromarray(test_image)


# Create image draw instance
draw = ImageDraw.Draw(pil_image)

# loop through faces in test image
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding )
  name = "Unknown person"

  if True in matches:
    first_match_index = matches.index(True)
    name = known_face_names[first_match_index]
  
  # Draw box
  draw.rectangle(((left,top), (right, bottom)), outline = (0,0,0))
  
  # Draw Label
  _, _, text_width, text_height = draw.textbbox((0,0),text = name)
  draw.rectangle (
    (
      (left, bottom - text_height - 10),
      (right, bottom)
    ),
    fill = (0,0,0),
    outline = (0,0,0)
    )
  draw.text((left+6, bottom-text_height - 5), name, fill=(255,255,255,255))
del draw
pil_image.show()

