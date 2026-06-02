from src.utils import load_hdf5_dataset

X, y, label_names = load_hdf5_dataset('dataset_colsign_45_154.h5')
# X: float32 array (N, 120, 258) — listo para LSTM
# y: int64 array (N,) — etiquetas como enteros
# label_names: list[str] — label_names[k] es el nombre de la clase k
print(X.shape)
print(y.shape)
print(len(label_names))
print(label_names[0])
# y luego, p.ej. con Keras:
# model.fit(X, y, validation_split=0.2, epochs=50, batch_size=32, ...)