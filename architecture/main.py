import getDataLabeled as gdl
import preprocess as p
import cnn
import predict


print("\n #### Get frames videos labeled ####")
frames_array_labeled = gdl.getFrames()
print(f"Frames shape for video : {frames_array_labeled.shape}")

print("\n #### Get angles videos labeled ####")
angles_array_labeled = gdl.getAngles()
print(f"Angles shape for video : {angles_array_labeled.shape}")

print("\n #### Replace NaN values by 0 and print dataType ####")
frames_array_labeled, angles_array_labeled = p.replaceNaN(frames_array_labeled, angles_array_labeled)

print("\n #### Init CNN ####")
model = None
model = cnn.CNNmodel(model)
#model = cnn.VGG16(model)

print("\n #### Train ####")
cnn.train(frames_array_labeled, angles_array_labeled, model)

print("\n #### Predict ####")
predict.prediction(model)


