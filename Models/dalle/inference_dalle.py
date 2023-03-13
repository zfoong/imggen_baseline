import tensorflow as tf
import numpy as np
from bpemb import BPEmb
from dalle_tensorflow.dalle_tensorflow import DALLE, DiscreteVAE
from tensorflow.keras.preprocessing.image import save_img


class inference_dalle:
    
    def __init__(self):
        
        print("Initializing Dall-E model...")
        
        # DiscreteVAE Params
        self.IMG_SIZE = 128
        NUM_VISUAL_TOKENS = 8192
        CODEBOOK_DIM = 512
        NUM_LAYERS = 2
        NUM_RESBLOCKS = 2
        HIDDEN_DIM = 256
        TEMPERATURE = 0.9
        STRAIGHT_THROUGH = False

        # Build our model and load DiscreteVAE with pre-trained weights
        vae = DiscreteVAE(
            image_size=self.IMG_SIZE,                # Size of image
            num_tokens=NUM_VISUAL_TOKENS,       # Number of visual tokens: The paper used 8192, but could be smaller for downsized projects
            codebook_dim=CODEBOOK_DIM,          # Codebook dimension
            num_layers=NUM_LAYERS,              # Number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
            num_resblocks=NUM_RESBLOCKS,        # Number of resnet blocks
            hidden_dim=HIDDEN_DIM,              # Hidden dimension
            temperature=TEMPERATURE,            # Gumbel softmax temperature. The lower this is, the harder the discretization
            straight_through=STRAIGHT_THROUGH   # Straight-through for gumbel softmax. unclear if it is better one way or the other
        )

        vae(image=tf.zeros(shape=[1, self.IMG_SIZE, self.IMG_SIZE, 3]))
        vae.load_weights("dalle_tensorflow/model_weights/vae/vae_weights")

        # DALLE Params
        self.VOCAB_SIZE = 10000
        assert self.VOCAB_SIZE in BPEmb.available_vocab_sizes('en'), "Vocab size not available. " \
                                                                "Call `BPEmb.available_vocab_sizes('en')` to check sizes"
        self.TEXT_SEQ_LEN = 128
        DEPTH = 16
        HEADS = 8
        DIM_HEAD = 64
        REVERSIBLE = True
        ATTN_DROPOUT = 0.1
        FF_DROPOUT = 0.1

        # Build our DALL-E model
        self.dalle = DALLE(
            dim=CODEBOOK_DIM,                           # Codebook Dimension
            vae=vae,                        # DiscreteVAE instance: image sequence length and number of image tokens inferred
            num_text_tokens=self.VOCAB_SIZE + 1,             # Vocab size for text. Add 1 for <PAD>
            text_sequence_len=self.TEXT_SEQ_LEN,        # Text sequence length
            depth=DEPTH,                                   # Transformer depth: should aim to be 64
            heads=HEADS,                                   # Attention heads
            dim_head=DIM_HEAD,                                # Attention head dimension
            reversible=REVERSIBLE,                            # Whether to use ReversibleSequence or SequentialSequence
            attn_dropout=ATTN_DROPOUT,                           # Attention dropout
            ff_dropout=FF_DROPOUT                              # Feedforward dropout
        )

        self.dalle.load_weights("dalle_tensorflow/model_weights/dalle/dalle_weights")
        
        print("Initialization completed")
    
    def inference(self, text):   
        print("Dall-E inferencing...")
        bpe_encoder = BPEmb(lang="en", vs=self.VOCAB_SIZE, add_pad_emb=True)
        text = bpe_encoder.encode_ids(text)
        text = np.array(text)
        text = np.pad(array=text, pad_width=[0, self.TEXT_SEQ_LEN - len(text)])
        text = tf.expand_dims(text, axis=0)
        mask = tf.cast(tf.where(text != 0, 1, 0, text), dtype=tf.bool)
        
        output_images = self.dalle.generate_images(text, mask=mask)
        output_images = tf.reshape(tensor=output_images, shape=[self.IMG_SIZE, self.IMG_SIZE, 3])
        output_images = save_img(path="dalle_out.jpg", x=output_images)
        print("Dall-E inferencing completed")
        return output_images

dalle = inference_dalle()
dalle.inference("a police sitting on a chair")