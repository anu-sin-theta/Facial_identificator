import face_recognition
#ye dlib ka error de sakti hai install karte time so be attentive.🫡
# Load images of known persons
person1_image = face_recognition.load_image_file("person1.jpg")
person2_image = face_recognition.load_image_file("person2.jpg")

person1_face_encoding = face_recognition.face_encodings(person1_image)[0]
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]

known_face_encodings = [person1_face_encoding, person2_face_encoding]
known_face_names = ["Person 1", "Person 2"]

unknown_image = face_recognition.load_image_file("test.jpg")

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for face_encoding, face_location in zip(face_encodings, face_locations):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = face_distances.argmin()
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    top, right, bottom, left = face_location
    cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(unknown_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

cv2.imshow("Recognized Faces", unknown_image)
cv2.waitKey(0)
cv2.destroyAllWindows()