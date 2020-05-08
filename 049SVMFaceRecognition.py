from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

faces = fetch_lfw_people(min_faces_per_person=60) # Getting faces

print(faces.target_names) # People who belong to this faces (names)
print(faces.images.shape) # shape dataset (images are 62*47 pixels)

fig, ax = plt.subplots(5, 6, figsize=(16,9))
for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(faces.images[i], cmap='bone')
    ax_i.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])
plt.show() # Displaying some faces

pca = PCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(faces.data, faces.target, random_state=42)

param_grid = {
    'svc__C': [0.1, 1, 5, 10, 50],
    'svc__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01]
}
grid = GridSearchCV(model, param_grid)
grid.fit(Xtrain, Ytrain)

print(f'Grid best params => {grid.best_params_}')

classifier = grid.best_estimator_
yfit = classifier.predict(Xtest)

fig, ax = plt.subplots(5, 6, figsize=(16, 9))
for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(Xtest[i].reshape(62,47), cmap='bone')
    ax_i.set(xticks=[], yticks=[])
    ax_i.set_ylabel(faces.target_names[yfit[i]].split()[-1], color='black' if yfit[i] == Ytest[i] else 'red')
fig.suptitle('Predicciones de las imagenes (incorrectas en rojo)')
plt.show()

print(classification_report(Ytest, yfit, target_names=faces.target_names))

mat = confusion_matrix(Ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True, xticklabels=faces.target_names, yticklabels=faces.target_names)
plt.show()