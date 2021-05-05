import streamlit as st
import tensorflow as tf


# Page
def write():
    """Used to write the page in the app.py file"""
    
    html_temp = """
    <div >
    <h1 style="color:rgb(191,0,0);text-align:center;">Classification bimodale </h1>
    </div>
    """
    html_temp1 = """
    <div >
    <h2 style="color:black;font-weight: bold;">Text encoder</h2>
    </div>
    """
    html_temp2 = """
    <div >
    <h2 style="color:black;font-weight: bold;">Image encoder</h2>
    </div>
    """
    html_temp3 = """
    <div >
    <h2 style="color:black;font-weight: bold;">Dual encoder</h2>
    </div>
    """
    html_temp4 = """
    <div >
    <h2 style="color:black;font-weight: bold;">Classification</h2>
    </div>
    """
    
    
    st.markdown(html_temp,unsafe_allow_html=True)
    st.markdown(html_temp1,unsafe_allow_html=True)
    text_part=st.beta_container()
    with st.echo():
         def text_bi_lstm(shape, embedding_dim,voc_size_inp,projection_dims, dropout_rate, training=False):
            
            inputs = Input(shape=shape, dtype='int32', name='text_input')
            text_embedding = Embedding(voc_size_inp, embedding_dim)(inputs)
            x = Bidirectional(LSTM(100, return_sequences=True))(text_embedding)
            x=Dropout(dropout_rate)(x)
            x = Bidirectional(LSTM(100))(x)
            x=Dropout(dropout_rate)(x)
           
            # Entrainement des top layers
            x=Flatten()(x)
            outputs = Dense(128,activation='relu')(x)
    
            # creation du text encoder model.
            model=Model(inputs, outputs, name="text_bi_lstm")
            model.summary()
            return model
    st.markdown(html_temp2,unsafe_allow_html=True)
    image_part=st.beta_container()
    with st.echo():
        def image_MobileNet(projection_dims, dropout_rate, trainable=False):
            # Load the pre-trained MobileNetV2 model to be used as the base encoder.
            MobileNetV2 = tf.keras.applications.MobileNetV2(
                include_top=False, weights="imagenet", pooling="avg"
            )
            # Set the trainability of the base encoder.
            for layer in MobileNetV2.layers:
            
                layer.trainable = trainable
            # Receive the images as inputs.
            inputs = Input(shape=(299, 299, 3), name="image_input")
            # Preprocess the input image.
            mobilenet_input = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
            
            # Generate the embeddings for the images using mobilenet transfer learning.
            embeddings = MobileNetV2(mobilenet_input)
            
            # Entrainement des top layers
            x = Dense(units=projection_dims,activation='relu')(embeddings)
            x = Dropout(dropout_rate)(x)
            outputs = Dense(128,activation='relu')(x)
        
            # Create the vision encoder model.
            model=Model(inputs, outputs, name="image_MobileNet_encoder")
            model.summary()
            return model
        
    st.markdown(html_temp3,unsafe_allow_html=True)
    bimodale_part=st.beta_container()
    
    with st.echo():
         class DualEncoder(tf.keras.Model):
          # Permet de hériter de la classe Model
            def __init__(self, text_encoder, image_encoder, **kwargs):
                
                super(DualEncoder, self).__init__(**kwargs)
                self.text_encoder = text_encoder
                self.image_encoder = image_encoder
                self.dual_dropout = tf.keras.layers.Dropout(0.3)
                self.dual_dense = tf.keras.layers.Dense(27, activation='softmax')
                self.loss_tracker = tf.keras.metrics.Mean(name="loss")
                self.f1_score=tfa.metrics.F1Score(num_classes=27, average='macro', name="f1_score")
                self.accuracy=tf.keras.metrics.CategoricalAccuracy(name='accuracy')
           
        
        
            @property 
            def metrics(self):
              # On liste les metriques utilisées pour que reset_states 
              # puisse être appelé automatiquement au début de chaque epoch 
              return [self.loss_tracker, self.accuracy, self.f1_score]
            
            
        
            #Dans call on définit les inputs des deux encoders , on appelle les deux encoders qui retournent des embeddings text et images
            # On applique un dual entrainement sur les deux embeddings et on output le résultat
            def call(self, features, training=False):
               # embeddings texte.
               text_embeddings = self.text_encoder(features["designation"], training=training)
               #embeddings  images.
               image_embeddings = self.image_encoder(features["image"], training=training)
               dual_embeddings = concatenate([image_embeddings,text_embeddings])
               
               x=self.dual_dropout(dual_embeddings)
               x=self.dual_dense(x)
               return x
        
            def loss_macro_soft_f1(self, y, y_hat):
              # Definition de la loss macro F1
              y = tf.cast(y, tf.float32)
              y_hat = tf.cast(y_hat, tf.float32)
              tp = tf.reduce_sum(y_hat * y, axis=0) # on multiplie la proba prédite d'un classe (y_hat) par son label=> Uniquement les proba des vrai positifs seront non nuls
              fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
              fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
              soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16) #  calcul du F1 score , 1e-16 pour ne pas diviser par 0
              cost = 1 - soft_f1 # comme on cherche a maximiser F1_score , et qu'il nous faut une fonction coût à minimiser duce 1 - soft-f1 in order to increase soft-f1
              macro_cost = tf.reduce_mean(cost) # on fait la moyenne pour tous les labels du batch
              return macro_cost
        
            def train_step(self, features):
                with tf.GradientTape() as tape:
                    # Forward pass
                    predictions = self(features, training=True)
                    loss = self.loss_macro_soft_f1(features['label'], predictions)
                # Backward pass
                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                # Monitor loss
                self.loss_tracker.update_state(loss)
                self.accuracy.update_state(features['label'], predictions)
                self.f1_score.update_state(features['label'], predictions)
                return {"loss": self.loss_tracker.result(),"accuracy": self.accuracy.result(),"f1_score": self.f1_score.result()}
        
            def test_step(self, features):
                x = self(features, training=False)
                loss = self.loss_macro_soft_f1(x, features['label'])
                self.loss_tracker.update_state(loss)
                self.accuracy.update_state(features['label'], x)
                self.f1_score.update_state(features['label'], x)
                return {"loss": self.loss_tracker.result(),"accuracy": self.accuracy.result(),"f1_score": self.f1_score.result()}
        
            def get_config(self):
                return {"text_encoder": self.text_encoder, "image_encoder": self.image_encoder}
        
            @classmethod
            def from_config(cls, config):
                return cls(**config)
