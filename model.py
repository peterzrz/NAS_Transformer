from keras.models import Model
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D
from transformer import Transformer



# generic model design
def model_fn(actions):
    # unpack the actions from the list
    kernel_1, filters_1, act_1, kernel_2, filters_2, kernel_3, filters_3, kernel_4, filters_4 = actions

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    model = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=7765,
        target_vocab_size=7010,
        dropout_rate=dropout_rate,
        activation_encoder=[],
        activation_decoder=[])
    return model
