import tensorflow as tf

#Model1 for Aspect Categorization - BiGRU
model1 = tf.keras.Sequential([
    tf.keras.layers.Embedding(2100,24, input_length=100),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(24,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(12)),
    tf.keras.layers.Dense(6,activation='relu'),
    tf.keras.layers.Dense(61,activation='softmax')
])

#Model2 for Sentiment Classification - BiGRU
model2 = tf.keras.Sequential([
    tf.keras.layers.Embedding(2100,8, input_length=100),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(8,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(4)),
    tf.keras.layers.Dense(4,activation='relu'),
    tf.keras.layers.Dense(4,activation='softmax')
])