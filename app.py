import tensorflow as tf

model = tf.keras.models.load_model("final_advanced_multi_domain_model.keras")

# Save in H5 legacy format (TensorFlow 2 compatible)
model.save("model_legacy.h5")